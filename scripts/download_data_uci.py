"""
Downloads raw data from `angeluriot/chess_games` Huggigface dataset. 
Outputs it to 'data/raw'

NOTE: This downloads (and caches) a 7.32 GB file (as of 2026-09-27)
Ensure sufficient disk space.
"""
from pathlib import Path
import sys
import math
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
import argparse
import math
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(
        description="Download and preprocess chess game data."
    )

    parser.add_argument(
        "num-games",
        type=int,
        help="Number of games to download and process.",
    )

    parser.add_argument(
        "train-fraction",
        type=float,
        help="Fraction of games assigned to the training set.",
    )

    # train-fraction is sufficient, this is just for extra protection from mistyping and such
    parser.add_argument(
        "val-fraction",
        type=float,
        help="Fraction of games assigned to the validation set.",
    )

    args = parser.parse_args()

    if not math.isclose(args.train_fraction + args.val_fraction, 1.0):
        parser.error("train_fraction and val_fraction must sum to 1.")

    if args.number_of_games <= 0:
        parser.error("number_of_games must be positive.")

    if not (0 <= args.train_fraction <= 1 and 0 <= args.val_fraction <= 1):
        parser.error("train_fraction and val_fraction must be between 0 and 1.")

    return args

args = parse_args()

N = args.number_of_games
train_fraction = args.train_fraction
val_fraction = args.val_fraction

RAW_DATA_DIR = Path("data/raw")

if not math.isclose(train_fraction + val_fraction, 1.0):
    print("Error: dataset fractions must sum to 1.")
    sys.exit(1)

huggingface_dataset = load_dataset("angeluriot/chess_games", split="train")
total = len(huggingface_dataset)
if N > total:
    print(f"Error: requested {N} games but dataset only has {total}.")
    sys.exit(1)


col = huggingface_dataset.data.table.column("moves_uci")
chunks = col.chunks
offsets = np.cumsum([0] + [len(c) for c in chunks])

schema = pa.schema([("moves_uci", pa.large_list(pa.large_string()))])

rng = np.random.default_rng(42)
perm = rng.permutation(total)
train_count = int(train_fraction * N)
val_count = N - train_count
date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
splits = {
    RAW_DATA_DIR / f"angeluriot_train_{date_str}.parquet":  perm[:train_count],
    RAW_DATA_DIR / f"angeluriot_val_{date_str}.parquet":    perm[train_count:train_count + val_count],
}

def gather(idx):
    chunk_ids = np.searchsorted(offsets, idx, side="right") - 1
    unique_ids = np.unique(chunk_ids)
    n = len(unique_ids)
    out_positions = []
    pieces = []
    for i, cid in enumerate(unique_ids):
        mask = chunk_ids == cid
        local = idx[mask] - offsets[cid]
        pieces.append(chunks[cid].take(pa.array(local)))
        out_positions.append(np.nonzero(mask)[0])
        print(f"\r  chunk {i + 1}/{n}", end="", flush=True)
    print("\r  reordering...        ", end="", flush=True)
    combined = pa.chunked_array(pieces)
    order = np.argsort(np.concatenate(out_positions))
    return combined.take(pa.array(order))

for path, idx in splits.items():
    with pq.ParquetWriter(path, schema, compression="zstd", compression_level=3) as writer:
        for start in range(0, len(idx), 200_000):
            part = gather(idx[start:start + 200_000])
            part = pa.table({"moves_uci": part}).cast(schema)
            writer.write_table(part)
            done = min(start + 200_000, len(idx))
            print(f"\r{path}: {done}/{len(idx)} rows ({100 * done / len(idx):.1f}%)          ", end="", flush=True)
    print()