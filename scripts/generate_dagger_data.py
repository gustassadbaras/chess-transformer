"""
Creates a new dataset containing model's games against stockfish, labelled by expert (stockfish), used for DAgger fine-tuning. 
NOTE: currently sampling opponent nodes unifiromly form [min_elo, max_elo], leads to strong-skewed opponent distirbution
fix with sampling in [log(min), log(max)]?
"""
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

from src.chess_transformer.dataset_manager import DatasetDirectory
from src.chess_transformer.checkpoint import load_checkpoint
from src.chess_transformer.simulator import make_engine_move, make_model_move
import config

def parse_args():
    parser = argparse.ArgumentParser(description="DAgger data generation")

    parser.add_argument(
        "checkpoint_path",
        type=Path,
    )
    parser.add_argument(
        "dataset_name",
        type=Path,
    )

    parser.add_argument("--num-games", type=int, default=5000)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--labeller-node-count", type=int, default=10000)
    parser.add_argument("--opponent-min-node-count", type=int, default=8000)
    parser.add_argument("--opponent-max-node-count", type=int, default=15000)

    return parser.parse_args()

args = parse_args()

CHECKPOINT_PATH = args.checkpoint_path
NUM_GAMES = args.num_games
NUM_WORKERS = args.num_workers
LABELLER_NODE_COUNT = args.labeller_node_count
OPPONENT_MIN_NODE_COUNT = args.opponent_min_node_count
OPPONENT_MAX_NODE_COUNT = args.opponent_min_node_count

DATASET_NAME = args.dataset_name
dataset_directory = DatasetDirectory(DATASET_NAME)
OUTPUT_PATH = dataset_directory.data_path()

dataset_directory.update_metadata(opponent_min_node_count=OPPONENT_MIN_NODE_COUNT,
                                  opponent_max_node_count=OPPONENT_MAX_NODE_COUNT,
                                  labeller_node_count=LABELLER_NODE_COUNT,
                                  min_opponent_node_count=OPPONENT_MIN_NODE_COUNT,
                                  max_opponent_node_count=OPPONENT_MAX_NODE_COUNT)

model_lock = threading.Lock()

# NOTE: could perhaps add support for labeller and opponent using different engines
def play_dagger_game(model, tokenizer, engine_path, engine_node_count):
    opponent = chess.engine.SimpleEngine.popen_uci(engine_path)
    labeler = chess.engine.SimpleEngine.popen_uci(engine_path)

    rows = []

    try:
        opponent.configure({
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
            make_engine_move(opponent, board, engine_node_count)

            if board.is_game_over():
                break

            best = labeler.play(board, chess.engine.Limit(nodes=LABELLER_NODE_COUNT)).move
            rows.append((board.fen(en_passant="fen"), best.uci()))

            model_move()
    finally:
        opponent.quit()
        labeler.quit()

    return rows


def worker(model, tokenizer):
    opponent_node_count = random.randint(OPPONENT_MIN_NODE_COUNT,
                                         OPPONENT_MAX_NODE_COUNT)
    return play_dagger_game(model, tokenizer, config.ENGINE_PATH, opponent_node_count)

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