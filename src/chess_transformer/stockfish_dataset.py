import torch
from torch.utils.data import Dataset
import pyarrow.parquet as pq
import numpy as np
import os

from .paths import get_path
from .simulator import encode_fen


class StockfishDataset(Dataset):
    def __init__(self, dataset_type, tokenizer):
        processed_path = get_path(dataset_type, "relabeled")
        self.table = pq.read_table(processed_path).to_pydict()
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.table)

    def __getitem__(self, i):
        encoded_board = encode_fen(self.table["fen"][i])
        encoded_label = self.tokenizer.output_token2id[self.table["best_uci"][i]]
        return torch.from_numpy(encoded_board).long(), torch.tensor(encoded_label)