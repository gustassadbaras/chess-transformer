"""
NOTE: This downloads (and caches) a 7.32 GB file (as of 2026-09-27)
Ensure sufficient disk space. 
"""
import torch
import math
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pyarrow.parquet as pq
import numpy as np

import config

class ChessDataset(Dataset):
    def __init__(self, uci_path, tokenizer, max_sequence_length):
        assert max_sequence_length % 3 == 1, (
            "max_sequence_length mod 3 must equal 1. "
            "(<|BOS|> + complete 3-token moves)."
        )
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length
        self.vocab_size = len(tokenizer.token2id)

        game_chunks = []
        game_lengths = []

        parquet_file = pq.ParquetFile(uci_path)
        metadata = pq.read_metadata(uci_path)
        num_games = metadata.num_rows
        total_batches = math.ceil(num_games / 10000)
        for batch_idx, batch in enumerate(
            parquet_file.iter_batches(columns=["moves_uci"], batch_size=10000)):
            for game_moves in batch.column("moves_uci").to_pylist():
                del game_moves[max_sequence_length:]
                game_tokens = [tokenizer.BOS_ID] + tokenizer.encode(game_moves)[:max_sequence_length]
                game_chunks.append(np.asarray(game_tokens, dtype=np.int32))
                game_lengths.append(len(game_tokens))
            print(
                f"\rLoading dataset consisting of {num_games} chess games. "
                f"Progress: {batch_idx+1}/{total_batches} "
                f"({(batch_idx+1) / total_batches:.1%})",
                end="",
                flush=True,
            )
        print("\nSuccessfully loaded dataset.")

        self.flat_tokens = np.concatenate(game_chunks)
        self.offsets = np.zeros(len(game_lengths) + 1, dtype=np.int64)
        np.cumsum(game_lengths, out=self.offsets[1:])

    def __len__(self):
        return len(self.offsets) - 1

    def __getitem__(self, idx):
        game_tokens = self.flat_tokens[self.offsets[idx]:self.offsets[idx+1]]
        return (
            torch.tensor(game_tokens[:-1], dtype=torch.long), 
            torch.tensor(game_tokens[1:], dtype=torch.long)
        )

def pad_collate_fn(batch):
    inputs, targets = zip(*batch)

    padded_inputs = pad_sequence(
        inputs,
        batch_first=True,
        padding_value=config.PAD_TOKEN_ID
    )

    padded_targets = pad_sequence(
        targets,
        batch_first=True,
        padding_value=config.PAD_TOKEN_ID
    )
    
    return padded_inputs, padded_targets