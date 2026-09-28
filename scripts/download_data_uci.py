# NOTE: this feels like it should be way faster. 
# This is quite complicated, I'm leaving it at this for now. 

import sys
import math
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
import config

if len(sys.argv) != 4:
    print("Usage: download_data_uci.py <number_of_games> <train_fraction> <val_fraction>")
    sys.exit(1)

N = int(sys.argv[1])
train_fraction, val_fraction = (float(a) for a in sys.argv[2:4])

if not math.isclose(train_fraction + val_fraction, 1.0):
    print("Error: dataset fractions must sum to 1.")
    sys.exit(1)

dataset = load_dataset("angeluriot/chess_games", split="train")
total = len(dataset)
if N > total:
    print(f"Error: requested {N} games but dataset only has {total}.")
    sys.exit(1)

col = dataset.data.table.column("moves_uci")
chunks = col.chunks
offsets = np.cumsum([0] + [len(c) for c in chunks])

schema = pa.schema([("moves_uci", pa.large_list(pa.large_string()))])

rng = np.random.default_rng(42)
perm = rng.permutation(total)
train_count = int(train_fraction * N)
val_count = N - train_count
splits = {
    config.UCI_TRAIN_DATA_PATH:   perm[:train_count],
    config.UCI_VAL_DATA_PATH:     perm[train_count:train_count + val_count],
    config.UCI_HOLDOUT_DATA_PATH: perm[train_count + val_count:],
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