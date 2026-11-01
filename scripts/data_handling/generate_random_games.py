import json
import random
import sys
import chess

import config

# Usage:
# generate_random_games.py 
# <number of games> <minimum number of moves per game><maximum number of moves per game> <random seed>
#
# Example:
# generate_random_games.py 1000 10 100 42
#
# Each move is sampled uniformly from the currently legal moves.
# Games end when the position is game-over or the maximum game length is reached.

try:
    num_games = int(sys.argv[1])
    min_game_length = int(sys.argv[2])
    max_game_length = int(sys.argv[3])
    random_seed = int(sys.argv[4])
except (IndexError, ValueError):
    print(
        "Usage: generate_random_games.py "
        "<number of games> <minimum number of moves> <maximum number of moves> <random seed>"
    )
    sys.exit(1)


if num_games <= 0:
    print("Number of games must be positive.")
    sys.exit(1)

if min_game_length <= 0:
    print("Maximum game length must be positive.")
    sys.exit(1)

if max_game_length <= 0:
    print("Maximum game length must be positive.")
    sys.exit(1)


random.seed(random_seed)

output_path = config.RANDOM_GAME_PATH

with open(output_path, "w", encoding="utf-8") as f:
    for _ in range(num_games):
        board = chess.Board()
        moves_uci = []

        game_length = random.randint(min_game_length, max_game_length)
        for _ in range(game_length):
            if board.is_game_over():
                break

            move = random.choice(list(board.legal_moves))

            moves_uci.append(move.uci())
            board.push(move)

        game = {
            "moves_uci": moves_uci
        }

        f.write(json.dumps(game) + "\n")


print(f"Generated {num_games} games.")
print(f"Maximum game length: {max_game_length} moves.")
print(f"Random seed: {random_seed}")
print(f"Saved to: {output_path}")