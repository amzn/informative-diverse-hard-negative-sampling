# This file was modified from the original Tevatron package.

import itertools
import os
import pickle
from itertools import repeat
from typing import Dict, List, Tuple, Optional, Any, Union

import pandas as pd
from tqdm import tqdm
from transformers.trainer import Trainer

import torch
from torch.utils.data import DataLoader, RandomSampler, Sampler
import torch.distributed as dist

from .loss import SimpleContrastiveLoss, DistributedContrastiveLoss

import logging

logger = logging.getLogger(__name__)

try:
    from grad_cache import GradCache

    _grad_cache_available = True
except ModuleNotFoundError:
    _grad_cache_available = False

RESOURCE_FOLDER = f"../resources"


class ClusterSampler(Sampler):
    def __init__(self, dataset, path, world_size=None, process_index=None):
        super(ClusterSampler, self).__init__(dataset)
        with open(path, "rb") as handle:
            clustered_batches = pickle.load(handle)
        qid_to_tok_hash = pd.read_parquet(f"{RESOURCE_FOLDER}/qid_to_tok_hash.parquet")
        qid_to_tok_hash = qid_to_tok_hash.to_dict()["tok_hash"]
        query_tok_hash_to_index = {
            hash(tuple(train_entry["query"][:32])): index
            for index, train_entry
            in enumerate(tqdm(dataset.train_data, desc="creating sampler"))
        }
        self.qids_per_epoch = [
            [query_tok_hash_to_index[qid_to_tok_hash[int(query)]] for query in itertools.chain(*epoch_seq)]
            for epoch_seq in clustered_batches
        ]
        self.epoch = -1
        self.world_size = world_size or 1
        self.process_index = process_index or 0
        self.padded_len = len(self.qids_per_epoch[0]) - len(self.qids_per_epoch[0]) % self.world_size

    def __len__(self):
        return self.padded_len

    def __iter__(self):
        self.epoch += 1
        print(f"Sample starts a new epoch {self.epoch}")
        curr_qids = self.qids_per_epoch[self.epoch % len(self.qids_per_epoch)]
        curr_qids = curr_qids[self.process_index:self.padded_len:self.world_size]
        yield from curr_qids


class TevatronTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super(TevatronTrainer, self).__init__(*args, **{k: v for k, v in kwargs.items() if k != "queries_batches_path"})
        self._dist_loss_scale_factor = dist.get_world_size() if self.args.negatives_x_device else 1
        self.queries_batches_path = kwargs.get("queries_batches_path", None)

    def _save(self, output_dir: Optional[str] = None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", output_dir)
        self.model.save(output_dir)

    def _prepare_inputs(
            self,
            inputs: Tuple[Dict[str, Union[torch.Tensor, Any]], ...]
    ) -> List[Dict[str, Union[torch.Tensor, Any]]]:
        prepared = []
        for x in inputs:
            if isinstance(x, torch.Tensor):
                prepared.append(x.to(self.args.device))
            else:
                prepared.append(super()._prepare_inputs(x))
        return prepared

    def _get_train_sampler(self):
        if self.queries_batches_path is not None:
            return ClusterSampler(self.train_dataset, self.queries_batches_path,
                                  self.args.world_size, self.args.process_index)
        return super()._get_train_sampler()

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        train_sampler = self._get_train_sampler()

        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=True,
            num_workers=self.args.dataloader_num_workers,
        )

    def compute_loss(self, model, inputs):
        query, passage = inputs
        return model(query=query, passage=passage).loss

    def training_step(self, *args):
        return super(TevatronTrainer, self).training_step(*args) / self._dist_loss_scale_factor


def split_dense_inputs(model_input: dict, chunk_size: int):
    assert len(model_input) == 1
    arg_key = list(model_input.keys())[0]
    arg_val = model_input[arg_key]

    keys = list(arg_val.keys())
    chunked_tensors = [arg_val[k].split(chunk_size, dim=0) for k in keys]
    chunked_arg_val = [dict(zip(kk, tt)) for kk, tt in zip(repeat(keys), zip(*chunked_tensors))]

    return [{arg_key: c} for c in chunked_arg_val]


def get_dense_rep(x):
    if x.q_reps is None:
        return x.p_reps
    else:
        return x.q_reps


class GCTrainer(TevatronTrainer):
    def __init__(self, *args, **kwargs):
        logger.info('Initializing Gradient Cache Trainer')
        if not _grad_cache_available:
            raise ValueError(
                'Grad Cache package not available. You can obtain it from https://github.com/luyug/GradCache.')
        super(GCTrainer, self).__init__(*args, **kwargs)

        loss_fn_cls = DistributedContrastiveLoss if self.args.negatives_x_device else SimpleContrastiveLoss
        loss_fn = loss_fn_cls()

        self.gc = GradCache(
            models=[self.model, self.model],
            chunk_sizes=[self.args.gc_q_chunk_size, self.args.gc_p_chunk_size],
            loss_fn=loss_fn,
            split_input_fn=split_dense_inputs,
            get_rep_fn=get_dense_rep,
            fp16=self.args.fp16,
            scaler=self.scaler if self.args.fp16 else None
        )

    def training_step(self, model, inputs) -> torch.Tensor:
        model.train()
        queries, passages = self._prepare_inputs(inputs)
        queries, passages = {'query': queries}, {'passage': passages}

        _distributed = self.args.local_rank > -1
        self.gc.models = [model, model]
        loss = self.gc(queries, passages, no_sync_except_last=_distributed)

        return loss / self._dist_loss_scale_factor
