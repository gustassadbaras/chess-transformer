"""
Compare two chess transformer models (or a model vs. Stockfish at a fixed node count).

Every opening is played twice, with colors swapped, so both players get both sides
of every opening.

Examples:
    python compare_models.py --name a_vs_b --model1 models/25112.pth --model2 models/20732.pth \
        --openings books/book.parquet --games 100 --seed 1

    python compare_models.py --name a_vs_sf --model1 models/25112.pth --stockfish-nodes 1000 \
        --stockfish-path C:/sf/stockfish.exe --openings books/book.parquet --games 200
"""
import argparse
import json
import math
import os
import platform
import random
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import chess
import chess.engine
import chess.pgn
import pandas as pd
import torch

from src.chess_transformer.checkpoint import load_checkpoint
from src.chess_transformer.simulator import make_model_move

OUTPUT_DIR = Path("comparisons")

class ModelPlayer:
    def __init__(self, path, device):
        self.path = path
        self.name = Path(path).stem
        self.model, self.tokenizer, *_ = load_checkpoint(path, device)
        self.model.eval()
        params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Loaded {self.name} ({params:,} trainable params)")

    def start_game(self):
        pass

    def end_game(self):
        pass

    def move(self, board):
        with torch.no_grad():
            make_model_move(self.model, self.tokenizer, board)


class StockfishPlayer:
    def __init__(self, path, nodes, hash_mb):
        self.path = path
        self.nodes = nodes
        self.hash_mb = hash_mb
        self.name = f"Stockfish_{nodes}nodes"
        self.engine = None
        self.engine_id = None

    def start_game(self):
        # Fresh process per game => no hash / state carries over between games.
        self.engine = chess.engine.SimpleEngine.popen_uci(self.path)
        self.engine.configure({"Threads": 1, "Hash": self.hash_mb})
        self.engine_id = dict(self.engine.id)

    def end_game(self):
        if self.engine is not None:
            self.engine.quit()
            self.engine = None

    def move(self, board):
        result = self.engine.play(board, chess.engine.Limit(nodes=self.nodes))
        board.push(result.move)

def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np
        np.random.seed(seed % (2**32))
    except ImportError:
        pass

def play_game(white, black, fen, event, round_no):
    """Play one game from `fen`. Returns (result_str, termination_str, pgn_str, plies)."""
    board = chess.Board(fen)
    players = {chess.WHITE: white, chess.BLACK: black}
    forfeit_winner = None

    white.start_game()
    if black is not white:
        black.start_game()
    try:
        while True:
            outcome = board.outcome(claim_draw=True)
            if outcome is not None:
                break
            before = len(board.move_stack)
            players[board.turn].move(board)
            if len(board.move_stack) != before + 1:
                forfeit_winner = not board.turn  # mover failed to produce a move
                break
    finally:
        white.end_game()
        if black is not white:
            black.end_game()

    if forfeit_winner is not None:
        result = "1-0" if forfeit_winner == chess.WHITE else "0-1"
        termination = "ILLEGAL_OR_MISSING_MOVE"
    else:
        result = outcome.result()
        termination = outcome.termination.name

    game = chess.pgn.Game.from_board(board)
    game.headers["Event"] = event
    game.headers["Round"] = str(round_no)
    game.headers["White"] = white.name
    game.headers["Black"] = black.name
    game.headers["Result"] = result
    game.headers["Termination"] = termination
    exporter = chess.pgn.StringExporter(headers=True, variations=False, comments=False, columns=80)
    pgn = game.accept(exporter)
    return result, termination, pgn, len(board.move_stack)

def player1_score(result, p1_is_white):
    if result == "1/2-1/2":
        return 0.5
    white_won = result == "1-0"
    return 1.0 if white_won == p1_is_white else 0.0


def elo_from_p(p):
    if p <= 0.0 or p >= 1.0:
        return None
    return 400 * math.log10(p / (1 - p))

def compute_stats(games):
    n = len(games)
    wins = sum(g["p1_score"] == 1.0 for g in games)
    draws = sum(g["p1_score"] == 0.5 for g in games)
    losses = sum(g["p1_score"] == 0.0 for g in games)
    stats = {"games": n, "wins": wins, "draws": draws, "losses": losses}
    if n == 0:
        return stats

    score = wins + 0.5 * draws
    p = score / n
    stats["score"] = score
    stats["score_pct"] = 100 * p

    for color, label in ((True, "as_white"), (False, "as_black")):
        sub = [g for g in games if g["p1_is_white"] == color]
        stats[label] = {
            "games": len(sub),
            "wins": sum(g["p1_score"] == 1.0 for g in sub),
            "draws": sum(g["p1_score"] == 0.5 for g in sub),
            "losses": sum(g["p1_score"] == 0.0 for g in sub),
        }

    # Paired (same opening, colors swapped) statistics.
    by_opening = {}
    for g in games:
        by_opening.setdefault(g["opening_index"], []).append(g["p1_score"])
    pair_scores = [sum(v) for v in by_opening.values() if len(v) == 2]
    pentanomial = {str(s): pair_scores.count(s) for s in (0.0, 0.5, 1.0, 1.5, 2.0)}
    stats["pair_score_distribution"] = pentanomial

    elo = elo_from_p(p)
    stats["elo_diff"] = elo
    stats["elo_ci95"] = None
    if len(pair_scores) >= 2:
        xs = [s / 2 for s in pair_scores]
        mean = sum(xs) / len(xs)
        var = sum((x - mean) ** 2 for x in xs) / (len(xs) - 1)
        se = math.sqrt(var / len(xs))
        lo, hi = mean - 1.96 * se, mean + 1.96 * se
        stats["elo_ci95"] = [elo_from_p(lo) if 0 < lo < 1 else None,
                             elo_from_p(hi) if 0 < hi < 1 else None]
        stats["score_ci95_pct"] = [100 * max(lo, 0.0), 100 * min(hi, 1.0)]
        stats["elo_basis"] = "pair-based (pentanomial) standard error; complete pairs only"
    return stats

def print_stats(stats, p1, p2):
    print("\n" + "=" * 60)
    print(f"{p1}  vs  {p2}")
    print("=" * 60)
    if stats["games"] == 0:
        print("No games completed.")
        return
    print(f"Games: {stats['games']}   (W {stats['wins']} / D {stats['draws']} / L {stats['losses']})"
          f"   [from {p1}'s perspective]")
    print(f"Score: {stats['score']} / {stats['games']} = {stats['score_pct']:.1f}%")
    for label in ("as_white", "as_black"):
        s = stats[label]
        print(f"  {label}: W {s['wins']} / D {s['draws']} / L {s['losses']}  ({s['games']} games)")
    print(f"Pair score distribution (0, 0.5, 1, 1.5, 2): {stats['pair_score_distribution']}")
    elo = stats["elo_diff"]
    if elo is None:
        print("Elo difference: undefined (score is 0% or 100%)")
    else:
        print(f"Elo difference ({p1} - {p2}): {elo:+.1f}")
    ci = stats.get("elo_ci95")
    if ci:
        fmt = lambda v: "inf" if v is None else f"{v:+.1f}"
        print(f"  95% CI: [{fmt(ci[0])}, {fmt(ci[1])}]")

def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="Output JSON file name (saved in the hard-coded directory).")
    ap.add_argument("--model1", required=True, help="Path to model 1 checkpoint.")
    ap.add_argument("--model2", help="Path to model 2 checkpoint (mutually exclusive with --stockfish-nodes).")
    ap.add_argument("--stockfish-nodes", type=int, help="Use Stockfish as player 2 with this node limit.")
    ap.add_argument("--stockfish-path", default=None, help="Stockfish binary (default: 'stockfish' on PATH).")
    ap.add_argument("--stockfish-hash", type=int, default=16, help="Stockfish hash size in MB (default 16).")
    ap.add_argument("--openings", required=True, help=".parquet file with a 'fen' column.")
    ap.add_argument("--num-games", type=int, required=True, help="Total number of games to play (even; each opening is played twice).")
    ap.add_argument("--seed", type=int, default=0, help="Seed for opening sampling and per-game seeding.")
    ap.add_argument("--device", default=None, help="torch device (default: cuda if available else cpu).")
    args = ap.parse_args()

    if (args.model2 is None) == (args.stockfish_nodes is None):
        ap.error("Provide exactly one of --model2 or --stockfish-nodes.")
    if args.stockfish_nodes is not None and args.stockfish_nodes < 1:
        ap.error("--stockfish-nodes must be >= 1.")
    if args.num_games < 2 or args.num_games % 2:
        ap.error("--num-games must be an even number >= 2 (each opening is played with both colors).")
    return args


def main():
    args = parse_args()

    out_name = Path(args.name).name
    if not out_name.endswith(".json"):
        out_name += ".json"
    out_path = OUTPUT_DIR / out_name
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pgn_path = out_path.with_suffix(".pgn")
    for p in (out_path, pgn_path):
        if p.exists():
            sys.exit(f"Refusing to overwrite existing file: {p}")

    # Openings
    fens = pd.read_parquet(args.openings, columns=["fen"])["fen"].tolist()
    n_openings = args.num_games // 2
    if n_openings > len(fens):
        sys.exit(f"Need {n_openings} openings but the book only has {len(fens)}.")
    rng = random.Random(args.seed)
    opening_indices = rng.sample(range(len(fens)), n_openings)

    # Players
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    p1 = ModelPlayer(args.model1, device)
    if args.model2 is not None:
        p2 = ModelPlayer(args.model2, device)
    else:
        sf_path = args.stockfish_path or shutil.which("stockfish")
        if sf_path is None:
            sys.exit("Stockfish binary not found; pass --stockfish-path.")
        p2 = StockfishPlayer(sf_path, args.stockfish_nodes, args.stockfish_hash)
    if p1.name == p2.name:
        p2.name += "_2"  # keep PGN names distinct

    event = f"{p1.name} vs {p2.name}"
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "args": vars(args),
        "player1": {"name": p1.name, "type": "model", "path": p1.path},
        "player2": ({"name": p2.name, "type": "model", "path": p2.path} if isinstance(p2, ModelPlayer)
                    else {"name": p2.name, "type": "stockfish", "path": p2.path, "nodes": p2.nodes,
                          "threads": 1, "hash_mb": p2.hash_mb}),
        "device": str(device),
        "openings_file": args.openings,
        "openings_in_book": len(fens),
        "opening_indices": opening_indices,
        "adjudication": "none; draws claimed on threefold repetition / fifty-move rule",
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "python_chess": chess.__version__},
    }

    games = []

    def save(completed):
        report = {
            "completed": completed,
            "metadata": metadata,
            "stats": compute_stats(games),
            "games": games,
        }
        tmp = out_path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        os.replace(tmp, out_path)

        pgn_tmp = pgn_path.with_suffix(".pgn.tmp")
        with open(pgn_tmp, "w", encoding="utf-8", newline="\n") as f:
            for g in games:
                f.write(g["pgn"].strip() + "\n\n")
        os.replace(pgn_tmp, pgn_path)

    game_index = 0
    completed = False
    try:
        for k, op_idx in enumerate(opening_indices):
            fen = fens[op_idx]
            for p1_is_white in (True, False):
                seed_everything(args.seed + game_index)
                white, black = (p1, p2) if p1_is_white else (p2, p1)
                t0 = time.time()
                result, termination, pgn, plies = play_game(white, black, fen, event, game_index + 1)
                if isinstance(p2, StockfishPlayer) and "stockfish_id" not in metadata:
                    metadata["stockfish_id"] = p2.engine_id
                s = player1_score(result, p1_is_white)
                games.append({
                    "game_index": game_index,
                    "opening_index": op_idx,
                    "opening_fen": fen,
                    "white": white.name,
                    "black": black.name,
                    "p1_is_white": p1_is_white,
                    "result": result,
                    "p1_score": s,
                    "termination": termination,
                    "plies": plies,
                    "seed": args.seed + game_index,
                    "pgn": pgn,
                })
                st = compute_stats(games)
                print(f"[{game_index + 1}/{args.num_games}] {white.name} vs {black.name}: {result} "
                      f"({termination}, {plies} plies, {time.time() - t0:.1f}s) | "
                      f"{p1.name}: +{st['wins']} ={st['draws']} -{st['losses']}")
                game_index += 1
        completed = True
    except KeyboardInterrupt:
        print("\nInterrupted; saving partial results.")
    finally:
        save(completed)

    print_stats(compute_stats(games), p1.name, p2.name)
    print(f"\nSaved to {out_path} and {pgn_path}")


if __name__ == "__main__":
    main()