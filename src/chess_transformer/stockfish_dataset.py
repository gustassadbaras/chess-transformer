import torch
from torch.utils.data import Dataset
import pyarrow.parquet as pq

from .simulator import encode_fen

class StockfishDataset(Dataset):
    def __init__(self, parquet_path, tokenizer):
        table = pq.read_table(parquet_path, columns=["fen", "best_uci"], memory_map=True)
        self.fens = table.column("fen")
        self.moves = table.column("best_uci")
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.fens)

    def __getitem__(self, i):
        encoded_board = encode_fen(self.fens[i].as_py())
        encoded_label = self.tokenizer.output_token2id[self.moves[i].as_py()]
        return torch.from_numpy(encoded_board).long(), torch.tensor(encoded_label)