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



import pandas as pd
import numpy as np

RESOURCE_FOLDER = "../resources"

def mrr(x: pd.Series, k=10):
    if k is not None:
        x = x[:k]
    return 1 / (x.argmax() + 1) if 1 in x.values else 0


def eval_mrr(df, score_field):
    assert df.label.dtype == np.int64 and df.qid.dtype == np.int64 and df[score_field].dtype in [np.float64, np.float32]
    df = df.sort_values(["qid", score_field], ascending=False)
    mrr_df = df.groupby("qid").label.apply(mrr)
    return mrr_df.mean(), mrr_df


def eval_ranking7k(df_or_path):
    qrels = pd.read_csv(f"{RESOURCE_FOLDER}/qrels.dev7k.csv")
    if not isinstance(df_or_path, pd.DataFrame):
        dev_rank = pd.read_csv(df_or_path, sep="\t", names=["qid", "docid", "score"])
    else:
        dev_rank: pd.DataFrame = df_or_path
    dev_rank = dev_rank[dev_rank.qid.isin(pd.Index(qrels.qid))]
    assert dev_rank.qid.nunique() == qrels.qid.nunique()
    dev_rank = qrels.merge(dev_rank, on=["qid", "docid"], how="outer")
    dev_rank["score"] = dev_rank["score"].fillna(-1000).astype(np.float32)
    dev_rank["label"] = dev_rank["label"].fillna(0).astype(np.int64)
    return eval_mrr(dev_rank, "score")


def eval_recall(df_or_path, ks=(5, 20, 100)):
    qrels = pd.read_csv(f"{RESOURCE_FOLDER}/qrels.dev7k.csv")
    if not isinstance(df_or_path, pd.DataFrame):
        dev_rank = pd.read_csv(df_or_path, sep="\t", names=["qid", "docid", "score"])
    else:
        dev_rank: pd.DataFrame = df_or_path
    dev_rank = dev_rank[dev_rank.qid.isin(pd.Index(qrels.qid))]
    assert dev_rank.qid.nunique() == qrels.qid.nunique()
    df = dev_rank.merge(qrels[["qid", "docid", "label"]], on=["qid", "docid"], how="left")
    results = {k: float(df.groupby("qid").head(k).groupby("qid").label.any().mean())
               for k in ks}
    return results


if __name__ == "__main__":
    import sys

    mrr_score, mrr_score_per_qip = eval_ranking7k(sys.argv[1])
    print("MRR: ", mrr_score)
    recalls = eval_recall(sys.argv[1])
    print("Recall@K: ", recalls)
