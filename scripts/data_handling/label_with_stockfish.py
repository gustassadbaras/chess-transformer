"""
Turns 'raw' dataset (output of download_data_uci.py) into a stockfish-relabelled dataset inside data/datasets
Relabelling is done in partitions, so the process can be stopped and resumed seamlessly. 
The .parquet partitions can be easily combined into a single table. 
"""
import argparse
import math
import os
from functools import partial
from multiprocessing import Pool
from pathlib import Path
import chess
import chess.engine
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

import config
from src.chess_transformer.dataset_manager import DatasetDirectory
from src.chess_transformer.tokenizer import UCITokenizer

def parse_args():
    parser = argparse.ArgumentParser(
        description="Batched dataset relabeling using stockfish (and deduplication)"
    )
    parser.add_argument(
        "raw_dataset_path",
        type=Path,
        help="Path of the .parquet file with a 'moves_uci' column containing lists of chess moves starting from the initial board position. "
    )

    parser.add_argument(
        "--dataset-name",
        type=Path,
        default=Path("default-dataset")
    )

    parser.add_argument(
        "--deduplication-game-order-path",
        type=Path,
        default=None,
        help="Optional .npy file containing game permutation order"
    )

    parser.add_argument("--num-games", type=int, default=5_000)
    parser.add_argument("--node-count", type=int, default=10_000)
    parser.add_argument("--games-per-part", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()

args = parse_args()

RAW_DATASET_PATH = args.raw_dataset_path
DEDUPLICATION_GAME_ORDER_PATH = args.deduplication_game_order_path
NODE_COUNT = args.node_count
GAMES_PER_PART = args.games_per_part
SEED = args.seed
DATASET_NAME = args.dataset_name
NUM_GAMES = args.num_games

NUM_WORKERS = min(16, os.cpu_count() or 1)

tokenizer = UCITokenizer()

def atomic_save_npy(path, arr):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.save(f, arr)
    os.replace(tmp, path)

def work(part_id, flat_path, offsets_path, parts_dir, permutation_path):
    flat = np.load(flat_path, mmap_mode="r")
    offsets = np.load(offsets_path, mmap_mode="r")
    order = np.load(permutation_path, mmap_mode="r")
    games = order[part_id * GAMES_PER_PART:(part_id + 1) * GAMES_PER_PART]

    game_ids, plies, fens, bests, humans = [], [], [], [], []
    with chess.engine.SimpleEngine.popen_uci(config.ENGINE_PATH) as engine:
        engine.configure({"Threads": 1})  # single-thread + node limit => reproducible
        for g in games:
            g = int(g)
            board = chess.Board()
            start, end = int(offsets[g]), int(offsets[g + 1])
            for ply, k in enumerate(range(start, end)):
                human = tokenizer.output_id2token[int(flat[k])]
                # game=g makes python-chess send ucinewgame between games,
                # clearing the hash so results don't depend on the previous game.
                best = engine.play(
                    board, chess.engine.Limit(nodes=NODE_COUNT), game=g
                ).move

                game_ids.append(g)
                plies.append(ply)
                fens.append(board.fen(en_passant="fen"))
                bests.append(best.uci())
                humans.append(human)

                try:
                    board.push(chess.Move.from_uci(human)) # No legality checks for speed, dataset gaurantees legality
                except ValueError as e:
                    raise ValueError(
                        f"Illegal human move {human!r} in game {g} at ply {ply}"
                    ) from e

    table = pa.table({
        "game_idx": pa.array(game_ids, pa.int32()),
        "ply": pa.array(plies, pa.int16()),
        "fen": pa.array(fens, pa.string()),
        "best_uci": pa.array(bests, pa.string()),
        "human_uci": pa.array(humans, pa.string()),
    })
    final = Path(parts_dir) / f"part_{part_id}.parquet"
    tmp = final.with_name(final.name + ".tmp")
    pq.write_table(table, tmp)
    os.replace(tmp, final)

def build_cache(processed_path, flat_path, offsets_path, num_games):
    pf = pq.ParquetFile(processed_path)
    if pf.metadata.num_rows < num_games:
        raise ValueError(
            f"Dataset has {pf.metadata.num_rows} games, need at least {num_games}"
        )

    chunks, lengths = [], []
    with tqdm(total=num_games, desc="Tokenizing games", unit="game") as bar:
        for batch in pf.iter_batches(columns=["moves_uci"], batch_size=10_000):
            remaining = num_games - len(lengths)
            if remaining <= 0:
                break
            for game_moves in batch.column("moves_uci").to_pylist()[:remaining]:
                tokens = tokenizer.encode(game_moves)
                chunks.append(np.asarray(tokens, dtype=np.int32))
                lengths.append(len(tokens))
                bar.update(1)

    flat = np.concatenate(chunks)
    offsets = np.zeros(len(lengths) + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])

    atomic_save_npy(offsets_path, offsets)
    atomic_save_npy(flat_path, flat)


def main():
    dataset_directory = DatasetDirectory(DATASET_NAME)

    permutation_path = dataset_directory.cache_path("game_permutation_order.npy") # 'game permutation order' sounds.. odd
    flat_path = dataset_directory.cache_path("flat_relabel.npy")
    offsets_path = dataset_directory.cache_path("relabelling_offsets.npy")

    # Parts are per dataset_type so one split's results are never mistaken for another's.
    parts_dir = dataset_directory.partitions_path()

    if not permutation_path.is_file():
        if not DEDUPLICATION_GAME_ORDER_PATH:
            print("Couldn't find permutation file, building one")
            indices_to_omit = np.array([], dtype=int)
        else:
            print(
                f"Couldn't find permutation file, building one while "
                f"omitting games from '{DEDUPLICATION_GAME_ORDER_PATH}' game indices"
            )
            indices_to_omit = np.load(DEDUPLICATION_GAME_ORDER_PATH, mmap_mode="r")

        available_indices = np.setdiff1d(
            np.arange(NUM_GAMES),
            indices_to_omit
        )
        perm = np.random.default_rng(SEED).permutation(available_indices)

        atomic_save_npy(permutation_path, perm)
    order = np.load(permutation_path, mmap_mode="r")
    if len(order) != NUM_GAMES:
        raise ValueError(f"Permutation has {len(order)} entries, expected {NUM_GAMES}")

    if not (Path(flat_path).is_file() and Path(offsets_path).is_file()):
        print(f"Turning dataset into flat & offset arrays")
        build_cache(RAW_DATASET_PATH, flat_path, offsets_path, NUM_GAMES)

    if len(np.load(offsets_path, mmap_mode="r")) - 1 < NUM_GAMES:
        raise ValueError(
            f"Cached arrays hold fewer than {NUM_GAMES} games; "
            f"delete {flat_path} and {offsets_path} to rebuild."
        )

    num_parts = math.ceil(NUM_GAMES / GAMES_PER_PART)

    for stale in parts_dir.glob("*.tmp"):
        stale.unlink()
    done = {int(p.stem.rsplit("_", 1)[1]) for p in parts_dir.glob("part_*.parquet")}
    todo = [p for p in range(num_parts) if p not in done]
    print(f"relabeling {len(todo)} of {num_parts} parts ({len(done)} already done)")

    fn = partial(work, flat_path=str(flat_path), offsets_path=str(offsets_path),
                 parts_dir=str(parts_dir), permutation_path=str(permutation_path))
    with Pool(NUM_WORKERS) as pool:
        for _ in tqdm(pool.imap_unordered(fn, todo), total=num_parts, initial=len(done),
                      desc="Relabeling", unit="part"):
            pass

    print("successfully relabeled dataset")


if __name__ == "__main__":
    main()