import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
import chess
import chess.engine
import pandas as pd
import torch
from tqdm import tqdm
import argparse
from pathlib import Path

from src.chess_transformer.checkpoint import load_checkpoint
from src.chess_transformer.simulator import make_engine_move, make_model_move
import config

def parse_args():
    parser = argparse.ArgumentParser(description="DAgger data generation")

    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=Path("models/model.pth")
    )
    parser.add_argument(
        "--dagger-dir",
        type=Path,
        default=Path("data/dagger")
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Output parquet path (automatically generated if omitted)"
    )

    parser.add_argument("--num-games", type=int, default=5000)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--min-engine-elo", type=int, default=1400)
    parser.add_argument("--max-engine-elo", type=int, default=2600)
    parser.add_argument("--node-count", type=int, default=10000)
    parser.add_argument("--s-per-engine-move", type=float, default=0.05)

    return parser.parse_args()

args = parse_args()

CHECKPOINT_PATH = args.checkpoint_path
DAGGER_DIR = args.dagger_dir

num_files = sum(1 for f in DAGGER_DIR.iterdir() if f.is_file())

OUTPUT_PATH = args.output_path or DAGGER_DIR / f"dagger_{num_files}.parquet"

NUM_GAMES = args.num_games
NUM_WORKERS = args.num_workers
MIN_ENGINE_ELO = args.min_engine_elo
MAX_ENGINE_ELO = args.max_engine_elo
NODE_COUNT = args.node_count
S_PER_ENGINE_MOVE = args.s_per_engine_move

model_lock = threading.Lock()


def play_dagger_game(model, tokenizer, engine_path, elo, seconds_per_engine_move):
    opponent = chess.engine.SimpleEngine.popen_uci(engine_path)
    labeler = chess.engine.SimpleEngine.popen_uci(engine_path)

    rows = []

    try:
        opponent.configure({
            "UCI_LimitStrength": True,
            "UCI_Elo": elo,
            "Threads": 1,
        })
        labeler.configure({"Threads": 1})

        board = chess.Board()

        def model_move():
            with model_lock, torch.no_grad():
                make_model_move(model, tokenizer, board)

        if random.random() < 0.5:
            model_move()

        while not board.is_game_over():
            make_engine_move(opponent, board, seconds_per_engine_move)

            if board.is_game_over():
                break

            best = labeler.play(board, chess.engine.Limit(nodes=NODE_COUNT)).move
            rows.append((board.fen(en_passant="fen"), best.uci()))

            model_move()
    finally:
        opponent.quit()
        labeler.quit()

    return rows


def worker(model, tokenizer):
    elo = random.randint(MIN_ENGINE_ELO, MAX_ENGINE_ELO)
    return play_dagger_game(model, tokenizer, config.ENGINE_PATH, elo,
                            S_PER_ENGINE_MOVE)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, tokenizer, optimizer, checkpoint_epoch, global_step, val_loss = (
        load_checkpoint(CHECKPOINT_PATH, device))
    model.eval()

    all_rows = []
    failed = 0

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as pool:
        futures = [pool.submit(worker, model, tokenizer) for _ in range(NUM_GAMES)]

        with tqdm(total=NUM_GAMES, desc="Games", unit="game") as bar:
            for fut in as_completed(futures):
                try:
                    all_rows.extend(fut.result())
                except Exception as e:
                    failed += 1
                    tqdm.write(f"Game failed: {e!r}")
                bar.set_postfix(positions=len(all_rows), failed=failed)
                bar.update(1)

    df = pd.DataFrame(all_rows, columns=["fen", "best_uci"])
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"Saved {len(df)} positions from {NUM_GAMES - failed} games to {OUTPUT_PATH}")