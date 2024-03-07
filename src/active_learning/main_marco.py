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
import json
import transformers
from active_learning.build_train_hn import tokenize_tevatron, TokenizeHardNegativesParams
from active_learning.get_hard_negatives import HardNegativeSelectionParams, get_hard_negatives
from active_learning.random_batching import BatchingParams, select_batches
from active_learning.msMarcoEval import eval_ranking7k
from tevatron.arguments import ModelArguments, DataArguments, \
    TevatronTrainingArguments as TrainingArguments
from tevatron.driver.encode import encode_main
from tevatron.driver.train import train_main
from tevatron.faiss_retriever.__main__ import RetrieverParams, retrieve_main

LOCAL_TEMP_FOLDER = f"temp"
RESOURCE_FOLDER = f"../resources"


@dataclass
class AlTrainParams:
    hard_neg_file_id: str


def dump_pickle(content, path):
    with open(path, "wb") as handle:
        return pickle.dump(content, handle)


def evaluate_marco(model_dir):
    query_encode_model_args = ModelArguments(model_dir)
    query_encode_data_args = DataArguments(
        q_max_len=32, encode_is_qry=True,
        encode_in_path=f"{RESOURCE_FOLDER}/dev7k.query.json",
        encoded_save_path=f"{LOCAL_TEMP_FOLDER}/encoded_queries.pt")
    query_encode_training_args = TrainingArguments(f"{LOCAL_TEMP_FOLDER}/encode", fp16=True,
                                                   per_device_eval_batch_size=128)
    encode_main(query_encode_model_args, query_encode_data_args, query_encode_training_args)
    doc_encode_model_args = ModelArguments(model_dir)
    doc_encode_data_args = DataArguments(
        encode_in_path=f"{RESOURCE_FOLDER}/top100_docs.json",
        encoded_save_path=f"{LOCAL_TEMP_FOLDER}/encoded_passages.pt")
    doc_encode_training_args = TrainingArguments(f"{LOCAL_TEMP_FOLDER}/encode", fp16=True,
                                                 per_device_eval_batch_size=128)
    encode_main(doc_encode_model_args, doc_encode_data_args, doc_encode_training_args)
    retriever_args = RetrieverParams(query_reps=f"{LOCAL_TEMP_FOLDER}/encoded_queries.pt",
                                     passage_reps=f"{LOCAL_TEMP_FOLDER}/encoded_passages.pt",
                                     depth=10, batch_size=-1, save_text=True,
                                     save_ranking_to=f"{LOCAL_TEMP_FOLDER}/dev.rank.tsv")
    retrieve_main(retriever_args)
    mrr, mrr_per_qid = eval_ranking7k(f"{LOCAL_TEMP_FOLDER}/dev.rank.tsv")
    print(f"************************** MRR is {mrr:.5f}")


def main(batching_args: BatchingParams, hard_neg_args: HardNegativeSelectionParams):
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments

    batches = select_batches(batching_args)
    batching_args_file = f'{RESOURCE_FOLDER}/batches.pkl'
    dump_pickle(batches, batching_args_file)
    with open(f'{RESOURCE_FOLDER}/batches.param', "w") as handle:
        json.dump({'class': batching_args.__class__.__name__, 'args': batching_args.__dict__}, handle)

    hard_negatives = get_hard_negatives(hard_neg_args)
    hard_neg_file_id = "hard_negatives"
    hard_neg_args_file = f'{RESOURCE_FOLDER}/{hard_neg_file_id}.pkl'
    dump_pickle(hard_negatives, hard_neg_args_file)
    with open(f'{RESOURCE_FOLDER}/hard_negatives.param', "w") as handle:
        json.dump({'class': hard_neg_args.__class__.__name__, 'args': hard_neg_args.__dict__}, handle)

    tokenized_path = f"{LOCAL_TEMP_FOLDER}/tokens_{hard_neg_file_id}"
    tokenization_args = TokenizeHardNegativesParams(hard_neg_args_file, save_to=tokenized_path)
    tokenize_tevatron(tokenization_args)

    model_args = ModelArguments(model_name_or_path="Luyu/co-condenser-marco")
    data_args = DataArguments(train_dir=tokenized_path)
    model_dir = f"{LOCAL_TEMP_FOLDER}/out/trained"
    training_args = TrainingArguments(output_dir=model_dir,
                                      save_steps=60000, fp16=True, per_device_train_batch_size=8,
                                      learning_rate=5e-6, dataloader_num_workers=2,
                                      dont_shuffle_negatives=True, num_train_epochs=3,
                                      queries_batches_path=batching_args_file)
    train_main(model_args, data_args, training_args)
    evaluate_marco(model_dir)


if __name__ == '__main__':
    print('Parsing parameters.')
    parser = transformers.HfArgumentParser((BatchingParams, HardNegativeSelectionParams))
    batch_args, hn_args = parser.parse_args_into_dataclasses()
    batch_args: BatchingParams
    hn_args: HardNegativeSelectionParams

    print('Batching arguments:')
    print(batch_args)

    print('Hard negative arguments:')
    print(hn_args)

    print('Starting experiment.')
    main(batch_args, hn_args)
