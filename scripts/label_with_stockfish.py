"""
NOTE: this script undersamples (due to deduplication), use higher N than desired number of examples
"""
from multiprocessing import Pool
import numpy as np
import chess
import chess.engine
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
import sys
import os
import math

import config
from src.chess_transformer.paths import get_path
from src.chess_transformer.tokenizer import UCITokenizer
tokenizer = UCITokenizer()

dataset_type = sys.argv[1]
flat_cache_path = get_path(dataset_type, "cached", cache_type="flat")
offset_cache_path = get_path(dataset_type, "cached", cache_type="offset")

NODE_COUNT = 10_000
NUM_WORKERS = 16
N = 1e6
NUM_PARTS = 64
PER_PART = N // NUM_PARTS

def work(part_id):
    flat = np.load(flat_cache_path, mmap_mode="r")
    offsets = np.load(offset_cache_path, mmap_mode="r") 
    total = int(offsets[-1])
    lo = part_id * total // NUM_PARTS
    hi = (part_id + 1) * total // NUM_PARTS

    rng = np.random.default_rng(part_id)
    spots = lo + rng.choice(hi - lo, size=min(PER_PART, hi-lo), replace=False)

    games = np.searchsorted(offsets, spots, side="right") - 1
    plies = spots - offsets[games]

    engine = chess.engine.SimpleEngine.popen_uci(config.ENGINE_PATH)
    rows = []
    for spot, g, ply in zip(spots.tolist(), games.tolist(), plies.tolist()):
        board = chess.Board()
        for k in range(offsets[g], offsets[g] + ply):
            board.push(chess.Move.from_uci(tokenizer.output_id2token[flat[k]]))
        best = engine.play(board, chess.engine.Limit(nodes=NODE_COUNT)).move
        human = tokenizer.output_id2token[flat[spot]]
        rows.append((board.fen(en_passant="fen"), best.uci(), human))
    engine.quit()
    return rows

if __name__ == "__main__":

    processed_path = get_path(dataset_type, "processed")

    print(f"Turning {dataset_type} dataset into flat & offset arrays")
    game_chunks = []
    game_lengths = []

    parquet_file = pq.ParquetFile(processed_path)
    metadata = pq.read_metadata(processed_path)
    num_games = metadata.num_rows # NOTE: could load less, room for shaving a few minutes off the script
    total_batches = math.ceil(num_games / 10000)
    for batch_idx, batch in enumerate(
        parquet_file.iter_batches(columns=["moves_uci"], batch_size=10000)):
        for game_moves in batch.column("moves_uci").to_pylist():
            game_tokens = tokenizer.encode(game_moves)
            game_chunks.append(np.asarray(game_tokens, dtype=np.int32))
            game_lengths.append(len(game_tokens))
        print(
            f"\rLoading dataset consisting of {num_games} chess games. "
            f"Progress: {batch_idx+1}/{total_batches} "
            f"({(batch_idx+1) / total_batches:.1%})",
            end="",
            flush=True,
        )

    flat = np.concatenate(game_chunks)
    offsets = np.zeros(len(game_lengths) + 1, dtype=np.int64)
    np.cumsum(game_lengths, out=offsets[1:])


    np.save(flat_cache_path, flat)
    np.save(offset_cache_path, offsets)


    unique = {}
    with Pool(NUM_WORKERS) as pool:
        for rows in tqdm(pool.imap(work, range(NUM_PARTS)), total=NUM_PARTS):
            for row in rows:
                key = row[0].rsplit(" ", 2)[0]   # position only, no move counters
                if key not in unique:
                    unique[key] = row

    fens, best, human = zip(*unique.values())
    table = pa.table({
        "fen": fens,
        "best_uci": best,
        "human_uci": human
    })
    pq.write_table(
        table,
        get_path(dataset_type, "relabeled")
    )
    print("successfully relabeled dataset")

    os.remove(flat_cache_path)
    os.remove(offset_cache_path)
