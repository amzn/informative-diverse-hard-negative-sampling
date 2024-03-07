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



import argparse
import pickle
import random
from dataclasses import dataclass

RESOURCE_FOLDER = f"../resources"

@dataclass
class BatchingParams:
    epochs: int = 3
    batching_seed: int = 55
    batch_size: int = 8


def generate_random_epoch(args: BatchingParams, query_ids):
    all_ids = query_ids
    random.shuffle(all_ids)
    return [all_ids[x:min(x + args.batch_size, len(all_ids))] for x in range(0, len(all_ids), args.batch_size)]


def select_batches(args: BatchingParams):
    random.seed(args.batching_seed)
    # load all needed data
    with open(f"{RESOURCE_FOLDER}/encoding/query/s1_train.pt", "rb") as handle:
        embeddings, query_ids = pickle.load(handle)

    return [generate_random_epoch(args, query_ids) for _ in range(args.epochs)]


def save_similar_groups_and_batches(args: BatchingParams, batches):
    batches_outfile = RESOURCE_FOLDER + "/batches.pkl"
    pickle.dump(batches, open(batches_outfile, "wb"))


if __name__ == '__main__':
    # parse
    parser = argparse.ArgumentParser(description='Batch similar queries iteratively.')
    parser.add_argument('--batch_seed', type=int, default=55, help='Seed for random operations.')
    parser.add_argument('--batch_size', type=int, default=8, help='Number of queries in each batch.')
    parser.add_argument('--epochs', type=int, default=3, choices=[1, 2, 3, 4, 5], help='Number of epochs to generate.')
    args_global: BatchingParams = parser.parse_args()
    print("Starting the batching process with following args: ", args_global)

    batches_created = select_batches(args_global)
    save_similar_groups_and_batches(args_global, batches_created)
