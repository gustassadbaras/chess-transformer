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
from src.chess_transformer.paths import get_path
from src.chess_transformer.tokenizer import UCITokenizer

NODE_COUNT = 10_000
NUM_WORKERS = min(16, os.cpu_count() or 1)
GAMES_PER_PART = 100
SEED = 42

RELABELED_DIR = Path("data/relabeled")

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
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_type")
    ap.add_argument("--num-games", type=int, default=1_000_000)
    args = ap.parse_args()
    dataset_type, num_games = args.dataset_type, args.num_games
    permutation_path = RELABELED_DIR / f"game_permutation_order_{dataset_type}.npy"

    flat_path = get_path(dataset_type, "cached", cache_type="flat", postfix="_stockfish")
    offsets_path = get_path(dataset_type, "cached", cache_type="offset", postfix="_stockfish")
    processed_path = get_path(dataset_type, "processed")

    # Parts are per dataset_type so one split's results are never mistaken for another's.
    parts_dir = RELABELED_DIR / "parts" / dataset_type
    parts_dir.mkdir(parents=True, exist_ok=True)

    if not permutation_path.is_file():
        print("Couldn't find permutation file, building one")
        perm = np.random.default_rng(SEED).permutation(num_games)
        atomic_save_npy(permutation_path, perm)
    order = np.load(permutation_path, mmap_mode="r")
    if len(order) != num_games:
        raise ValueError(f"Permutation has {len(order)} entries, expected {num_games}")

    if not (Path(flat_path).is_file() and Path(offsets_path).is_file()):
        if len(np.load(offsets_path, mmap_mode="r")) - 1 < num_games:
            raise ValueError(
                f"Cached arrays hold fewer than {num_games} games; "
                f"delete {flat_path} and {offsets_path} to rebuild."
            )
        print(f"Turning {dataset_type} dataset into flat & offset arrays")
        build_cache(processed_path, flat_path, offsets_path)

    num_parts = math.ceil(num_games / GAMES_PER_PART)

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