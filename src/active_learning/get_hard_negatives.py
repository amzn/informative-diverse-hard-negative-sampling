'''
  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

  Licensed under the Apache License, Version 2.0 (the "License").
  You may not use this file except in compliance with the License.
  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
'''



import itertools
import pickle
import faiss
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from glob import glob
import dask.dataframe as dd
from dataclasses import dataclass
import random
from multiprocess.pool import ThreadPool

RESOURCE_FOLDER = "../resources"

# Utils

def load_pickle(path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


@dataclass
class HardNegativeSelectionParams:
    num_of_starting_top_de_docs: int = 200
    ce_max: float = 0.6
    ce_min: float = 0.1
    ce_pos_thresh: float = 0.95  # threshold if we don't have enough samples based on ce_max
    k: int = 7
    negs_to_centers_ratio: int = 4
    num_of_epochs: int = 3
    active_learning_alg: str = "badge"
    should_filter_by_ce: bool = False
    should_apply_active_learning: bool = False
    should_dedup_neighbors_negatives: bool = False
    seed: int = 42
    batch_file_name: str = 'batches.pkl'
    dist_query_pos_choose_random: int = 5  # if query@pos < (query@negs).mean() + dist, then select randomly


def load_data(args: HardNegativeSelectionParams):
    encoding_path = f"{RESOURCE_FOLDER}/encoding"
    doc_embedding_files = sorted(glob(f"{encoding_path}/corpus/s1_split*"))
    query_enc_path = f"{encoding_path}/query/s1_train.pt"

    with ThreadPool(processes=len(doc_embedding_files) + 1) as pool:
        query_embes_aync_res = pool.apply_async(load_pickle, [query_enc_path])
        all_embeddings = pool.map(load_pickle, doc_embedding_files)
        query_embes = query_embes_aync_res.get()
    all_embeddings = np.concatenate([x[0] for x in all_embeddings])
    print("completed uploading doc + query embedding files")

    query_embedding_dict = {int(qid): embds for embds, qid in zip(query_embes[0], query_embes[1])}
    del query_embes

    qrels_path = f"{RESOURCE_FOLDER}/qrels.train.tsv"
    qrels = pd.read_csv(qrels_path, sep="\t", names=["qid", 0, "docid", "label"], usecols=["qid", "docid", "label"])
    query_pos_pairs = qrels.sample(frac=1, random_state=42).drop_duplicates(["qid"])
    print("completed uploading qrels")
    query_pos_pairs = {row[1]['qid']: row[1]['docid'] for row in query_pos_pairs.iterrows()}

    scores_path = f"{RESOURCE_FOLDER}/scores/*.parquet"

    ranking = dd.read_parquet(scores_path).compute()
    ranking = ranking.merge(qrels[["qid", "docid", "label"]], on=["qid", "docid"], how="left")
    ranking['label'] = ranking['label'].fillna(0)
    ranking = ranking.query('label == 0')
    ranking = {qid: group for qid, group in ranking.groupby('qid')}

    if args.batch_file_name:
        batches = load_pickle(f'{RESOURCE_FOLDER}/{args.batch_file_name}')
        assert len(batches) == args.num_of_epochs
    else:
        batches = None

    return all_embeddings, query_embedding_dict, query_pos_pairs, ranking, batches


def get_batch_mapping(batches):
    if batches:
        mapping = {}
        for i in range(len(batches)):
            mapping[i] = {}
            for batch in batches[i]:
                batchi = list(map(int, batch))
                for s in batchi:
                    mapping[i][s] = batchi

        return mapping
    else:
        return None


# Embedding functions

def batch_doc_independent_grad_embeddings(queries, pos, docs, args):
    queries.requires_grad_(False)
    pos.requires_grad_(False)

    neg_sim = queries @ docs.T
    pos_sim = (queries * pos).sum(axis=1).repeat(len(docs), 1).T
    if args.dist_query_pos_choose_random is not None:
        if pos_sim.mean().item() < neg_sim.mean().item() - args.dist_query_pos_choose_random:
            return None

    loss_per_doc = -(1 / (1 + (neg_sim - pos_sim).exp())).log()
    loss_per_doc.sum().backward()

    return docs.grad


# Clustering

def cluster_vectors(vecs, vec_ids, k=25, niter=10, nredo=1, seed=42):
    for i in range(10):
        kmeans = faiss.Kmeans(d=vecs.shape[1], k=k, niter=niter, nredo=nredo, min_points_per_centroid=1, seed=seed + i)
        kmeans.train(vecs)

        centroids = kmeans.centroids

        index = faiss.IndexFlatL2(vecs.shape[1])
        index.add(vecs)
        res = vec_ids[index.search(centroids, k=1)[1].squeeze()]
        if np.unique(res).size == k:
            return res.tolist()

    return res.tolist()


def run_embedding_al(all_embeddings, query_embedding_dict, query_pos_pairs, hard_negatives, k, n_epochs, qids, seed,
                     args):
    gradient_embds = 0
    num_of_contributing_queries = 0
    for qid in qids:
        cur_gradient_embds = get_embeddings(qid, all_embeddings, hard_negatives, query_embedding_dict,
                                            query_pos_pairs, args)
        if cur_gradient_embds is not None:
            num_of_contributing_queries += 1
            gradient_embds += cur_gradient_embds
    if type(gradient_embds) != int:
        gradient_embds /= num_of_contributing_queries
    else:
        gradient_embds = None

    if gradient_embds is None:
        selected_negs = random.choices(hard_negatives, k=n_epochs * k * len(qids))
    else:
        selected_negs = cluster_vectors(gradient_embds.numpy(), hard_negatives, n_epochs * k * len(qids), seed=seed)
        random.shuffle(selected_negs)
        selected_negs = list(selected_negs)
    return selected_negs


def get_embeddings(qid, all_embeddings, hard_negatives, query_embedding_dict, query_pos_pairs, args):
    if args.active_learning_alg == 'badge':
        negative_embeddings = torch.tensor(all_embeddings[hard_negatives], requires_grad=True)
        query_embeddings = torch.tensor([query_embedding_dict[qid]])
        query_embeddings.requires_grad = True
        pos_embeddings = torch.tensor(all_embeddings[[query_pos_pairs[qid]]], requires_grad=True)
        return batch_doc_independent_grad_embeddings(query_embeddings, pos_embeddings, negative_embeddings, args)
    elif args.active_learning_alg == 'bert-km':
        return torch.tensor(all_embeddings[hard_negatives])
    else:
        raise Exception('Illegal embedding based algorithm name!')


def remove_neighbours_negatives(batch_mapping, k, n_epochs, negative_rankings, qid, results, args):
    neighbors_per_epoch = [batch_mapping[e][qid] for e in range(n_epochs)]
    neighbors_negatives = []
    for e in range(n_epochs):
        if args.should_apply_active_learning:
            neighbors_negatives.extend(list(itertools.chain(
                *[results[i]['hard_neg'][e * k:(e + 1) * k] for i in neighbors_per_epoch[e] if i in results])))
        else:
            neighbors_negatives.extend(list(
                itertools.chain(*[results[i][e * k:(e + 1) * k] for i in neighbors_per_epoch[e] if i in results])))
    neighbors_negatives = set(neighbors_negatives)
    negative_rankings = negative_rankings[~negative_rankings['docid'].isin(neighbors_negatives)]
    return negative_rankings


def filter_by_ce(negative_rankings, ce_max, ce_min, ce_pt, k, n_epochs, negs_to_centers_ratio, batch_size=1):
    min_hard_negatives = int(n_epochs * k * negs_to_centers_ratio * batch_size)
    hard_negatives = negative_rankings.query(f"ce_score <= {ce_max} & ce_score >= {ce_min}")
    if hard_negatives.shape[0] < min_hard_negatives:
        hard_negatives = negative_rankings.query(f"ce_score <= {ce_pt} & ce_score >= {ce_min}")
        if hard_negatives.shape[0] < min_hard_negatives:
            hard_negatives = negative_rankings.query(f"ce_score <= {ce_pt}")
            hard_negatives = hard_negatives.nlargest(n=min_hard_negatives, columns='ce_score')
        else:
            hard_negatives = hard_negatives.nsmallest(n=min_hard_negatives, columns='ce_score')

    if len(hard_negatives) < min_hard_negatives:
        hard_negatives = hard_negatives.sample(n=min_hard_negatives, replace=True)

    return hard_negatives


def get_hard_negatives(args: HardNegativeSelectionParams):
    all_embeddings, query_embedding_dict, query_pos_pairs, ranking, batches = load_data(args)

    ce_max = args.ce_max
    ce_min = args.ce_min
    ce_pt = args.ce_pos_thresh
    k = args.k
    n_epochs = args.num_of_epochs
    if args.should_apply_active_learning and args.active_learning_alg == 'badge':
        negs_to_centers_ratio = args.negs_to_centers_ratio
    else:
        negs_to_centers_ratio = 1

    seed = args.seed
    random.seed(seed)

    batch_mapping = get_batch_mapping(batches)

    results = {}
    for qid in tqdm(ranking):
        negative_rankings = ranking[qid].nlargest(n=args.num_of_starting_top_de_docs, columns='de_score')

        if batch_mapping and args.should_dedup_neighbors_negatives:
            negative_rankings = remove_neighbours_negatives(batch_mapping, k, n_epochs, negative_rankings, qid, results,
                                                            args)

        if args.should_filter_by_ce:
            hard_negatives = filter_by_ce(negative_rankings, ce_max, ce_min, ce_pt, k, n_epochs, negs_to_centers_ratio)
        else:
            hard_negatives = negative_rankings

        if args.should_apply_active_learning:
            if args.active_learning_alg == 'badge' or args.active_learning_alg == 'bert-km':
                hard_negatives = hard_negatives['docid'].values
                selected_negs = run_embedding_al(all_embeddings, query_embedding_dict, query_pos_pairs, hard_negatives,
                                                 k, n_epochs, [qid], seed, args)
            elif args.active_learning_alg == 'uncertainty':
                selected_negs = hard_negatives.nlargest(n=k * n_epochs, columns='de_score')['docid'].values.tolist()
                random.shuffle(selected_negs)
            else:
                raise Exception('Illegal AL algorithm selected!')

            results[qid] = {'hard_neg': selected_negs}
        else:
            results[qid] = random.sample(hard_negatives['docid'].values.tolist(), k * n_epochs)

    return results


def get_hard_negatives_main(args: HardNegativeSelectionParams):
    results = get_hard_negatives(args)
    outfile = f"{RESOURCE_FOLDER}/hard_negatives.pkl"
    with open(outfile, "wb") as handle:
        pickle.dump(results, handle)
