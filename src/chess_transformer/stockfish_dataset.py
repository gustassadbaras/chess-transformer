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
        flat_cache_path = get_path(dataset_type, "cached", cache_type="flat")
        offset_cache_path = get_path(dataset_type, "cached", cache_type="offset")
        if os.path.exists(flat_cache_path) and os.path.exists(offset_cache_path):
            print(f"Loading cached dataset ({dataset_type})")
            self.tokens = np.load(flat_cache_path)
            self.labels = np.load(offset_cache_path)
        else:
            print(f"Loading dataset ({dataset_type}) for the first time")
            table = pq.read_table(processed_path).to_pydict()
            self.tokens = np.stack([encode_fen(f) for f in table["fen"]])
            self.labels = np.array([tokenizer.output_token2id[m] for m in table["best_uci"]], dtype=np.int64)

            np.save(flat_cache_path, self.tokens)
            np.save(offset_cache_path, self.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return torch.from_numpy(self.tokens[i]).long(), torch.tensor(self.labels[i])