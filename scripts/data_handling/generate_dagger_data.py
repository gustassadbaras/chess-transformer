"""
Creates a new dataset containing model's games against stockfish, labelled by expert (stockfish), used for DAgger fine-tuning. 

Example:
    python -m scripts.data_handling.generate_dagger_data "models/model.pth" "dagger_v1" "data/raw/openings_fen.parquet" --num-games 10000  
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
import math

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

    parser.add_argument(
        "openings_path",
        type=Path,
        help="Parquet file containing opening FENs"
    )

    parser.add_argument("--num-games", type=int, default=5000)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--labeller-nodes", type=int, default=10000)
    parser.add_argument("--opponent-min-nodes", type=int, default=100)
    parser.add_argument("--opponent-max-nodes", type=int, default=2000)

    return parser.parse_args()

args = parse_args()

CHECKPOINT_PATH = args.checkpoint_path
OPENINGS_PATH = args.openings_path
NUM_GAMES = args.num_games
NUM_WORKERS = args.num_workers
LABELLER_NODE_COUNT = args.labeller_nodes
OPPONENT_MIN_NODE_COUNT = args.opponent_min_nodes
OPPONENT_MAX_NODE_COUNT = args.opponent_max_nodes

DATASET_NAME = args.dataset_name
dataset_directory = DatasetDirectory(DATASET_NAME)
OUTPUT_PATH = dataset_directory.data_path()

dataset_directory.update_metadata(opponent_min_node_count=OPPONENT_MIN_NODE_COUNT,
                                  opponent_max_node_count=OPPONENT_MAX_NODE_COUNT,
                                  labeller_node_count=LABELLER_NODE_COUNT,
                                  pretrained_model_checkpoint_path=CHECKPOINT_PATH,
                                  openings_path=OPENINGS_PATH,
                                  type="dagger")

model_lock = threading.Lock()

# NOTE: could perhaps add support for labeller and opponent using different engines
def play_dagger_game(model, tokenizer, engine_path, engine_node_count, opening_fen):
    opponent = chess.engine.SimpleEngine.popen_uci(engine_path)
    labeler = chess.engine.SimpleEngine.popen_uci(engine_path)

    rows = []

    try:
        opponent.configure({
            "Threads": 1,
        })
        labeler.configure({"Threads": 1})

        board = chess.Board(opening_fen)

        def model_move():
            with model_lock, torch.no_grad():
                return make_model_move(model, tokenizer, board)

        def make_model_move_and_record():
            fen = board.fen(en_passant="fen")

            best = labeler.play(
                board,
                chess.engine.Limit(nodes=LABELLER_NODE_COUNT)
            ).move

            model_move_uci = model_move()

            rows.append((fen, best.uci(), model_move_uci))


        if random.random() < 0.5:
            make_model_move_and_record()

        while not board.is_game_over():
            make_engine_move(opponent, board, engine_node_count)

            if board.is_game_over():
                break

            make_model_move_and_record()
    finally:
        opponent.quit()
        labeler.quit()

    return rows

def worker(model, tokenizer, opening_fen):
    opponent_node_count = int(round(
        math.exp(random.uniform(
            math.log(OPPONENT_MIN_NODE_COUNT), 
            math.log(OPPONENT_MAX_NODE_COUNT)
            ))
        ))
    return play_dagger_game(model,
                            tokenizer, 
                            config.ENGINE_PATH, 
                            opponent_node_count,
                            opening_fen)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, tokenizer, optimizer, checkpoint_epoch, global_step, val_loss = (
        load_checkpoint(CHECKPOINT_PATH, device))
    model.eval()

    all_rows = []
    failed = 0

    opening_df = pd.read_parquet(OPENINGS_PATH)

    if "fen" not in opening_df.columns:
        raise ValueError("Opening dataset must contain a 'fen' column.")

    # probably unecessary
    opening_fens = (
        opening_df["fen"]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
    )

    # Online opening books exist, featuring
    # tens of thousands of openings, so this should be fine
    if len(opening_fens) < NUM_GAMES:
        raise ValueError(
            f"Only {len(opening_fens)} unique openings available "
            f"for {NUM_GAMES} requested games."
        )

    # Sample without replacement
    sampled_openings = random.sample(opening_fens, NUM_GAMES)

    print(f"Loaded {len(opening_fens)} unique openings.")
    print(f"Sampled {NUM_GAMES} openings.")

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as pool:
        futures = [pool.submit(worker,
                                model, 
                                tokenizer,
                                opening_fen
                                ) for opening_fen in sampled_openings]

        with tqdm(total=NUM_GAMES, desc="Games", unit="game") as bar:
            for fut in as_completed(futures):
                try:
                    all_rows.extend(fut.result())
                except Exception as e:
                    failed += 1
                    tqdm.write(f"Game failed: {e!r}")
                bar.set_postfix(positions=len(all_rows), failed=failed)
                bar.update(1)

    df = pd.DataFrame(all_rows, columns=["fen", "best_uci", "model_uci"])
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"Saved {len(df)} positions from {NUM_GAMES - failed} games to {OUTPUT_PATH}")