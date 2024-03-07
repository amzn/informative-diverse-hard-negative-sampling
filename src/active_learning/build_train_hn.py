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



import pickle
from dataclasses import dataclass
from transformers import AutoTokenizer
import os
import random
from tqdm import tqdm
from multiprocessing import Pool
from tevatron.preprocessor import MarcoPassageTrainPreProcessor as TrainPreProcessor

RESOURCE_FOLDER = '../resources'


@dataclass
class TokenizeHardNegativesParams:
    hard_negative_path: str  # path to the hard negative file (21 "interesting" negatives per query)
    hn_file: str = f"{RESOURCE_FOLDER}/s1_train.rank.tsv"  # contains top 200 docs for each query
    qrels: str = f"{RESOURCE_FOLDER}/qrels.train.tsv"
    queries: str = f"{RESOURCE_FOLDER}/train.query.txt"
    collection: str = f"{RESOURCE_FOLDER}/corpus.tsv"
    save_to: str = f"{RESOURCE_FOLDER}//tokenized_train_data"  # output is written here
    tokenizer_name: str = "bert-base-uncased"
    truncate: int = 128
    depth: int = 200
    mp_chunk_size: int = 500
    shard_size: int = 45000


def load_ranking(rank_file, relevance, depth, negatives_mapping):
    with open(rank_file) as rf:
        lines = iter(rf)
        q_0, p_0, _ = next(lines).strip().split()

        curr_q = q_0
        negatives = [] if p_0 in relevance[q_0] else [p_0]

        while True:
            try:
                q, p, _ = next(lines).strip().split()
                if q != curr_q:
                    negatives = negatives[:depth]
                    random.shuffle(negatives)
                    yield curr_q, relevance[curr_q], negatives_mapping[int(curr_q)]
                    curr_q = q
                    negatives = [] if p in relevance[q] else [p]
                else:
                    if p not in relevance[q]:
                        negatives.append(p)
            except StopIteration:
                negatives = negatives[:depth]
                random.shuffle(negatives)
                yield curr_q, relevance[curr_q], negatives_mapping[int(curr_q)]
                return


def tokenize_tevatron(args: TokenizeHardNegativesParams):
    qrel = TrainPreProcessor.read_qrel(args.qrels)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)
    processor = TrainPreProcessor(
        query_file=args.queries,
        collection_file=args.collection,
        tokenizer=tokenizer,
        max_length=args.truncate,
    )

    counter = 0
    shard_id = 0
    f = None
    os.makedirs(args.save_to, exist_ok=True)
    with open(args.hard_negative_path, "rb") as handle:
        negatives_mapping = pickle.load(handle)
    for k, v in negatives_mapping.items():
        if isinstance(v, dict):
            negatives_mapping[k] = v["hard_neg"]
    pbar = tqdm(load_ranking(args.hn_file, qrel, args.depth, negatives_mapping))
    with Pool() as p:
        for x in p.imap(processor.process_one, pbar, chunksize=args.mp_chunk_size):
            counter += 1
            if f is None:
                f = open(os.path.join(args.save_to, f'split{shard_id:02d}.hn.json'), 'w')
                pbar.set_description(f'split - {shard_id:02d}')
            f.write(x + '\n')

            if counter == args.shard_size:
                f.close()
                f = None
                shard_id += 1
                counter = 0

    if f is not None:
        f.close()
