import chess
import chess.pgn
import pandas as pd
import argparse

def convert(pgn_path, parquet_path, max_plies=16):
    rows = []

    with open(pgn_path, encoding="utf-8", errors="replace") as pgn:
        while game := chess.pgn.read_game(pgn):
            if game.errors:
                continue

            board = game.board()

            for move in list(game.mainline_moves())[:max_plies]:
                board.push(move)

            if board.ply() > max_plies:
                continue

            rows.append({
                "fen": board.fen(),
                "opening": game.headers.get("Opening", ""),
                "eco": game.headers.get("ECO", ""),
            })

    df = pd.DataFrame(rows).drop_duplicates(subset=["fen"])
    df.to_parquet(parquet_path, index=False)

    print(f"Saved {len(df):,} positions to {parquet_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pgn")
    parser.add_argument("output")
    parser.add_argument("--max-plies", type=int, default=1000)

    args = parser.parse_args()
    convert(args.pgn, args.output, args.max_plies)
