import random
import torch
import json

from .simulator import simulate_game, get_legal_token_ids, construct_board_from_token_ids
import config

def estimate_winrate(model, tokenizer, engine_path, engine_elo, seconds_per_engine_move):
    wins = 0

    for _ in range(config.GAMES_PER_WINRATE_ESTIMATE):
        _, outcome = simulate_game(model,
                            tokenizer,
                            engine_path, 
                            engine_elo,
                            seconds_per_engine_move)

        wins += 1 if outcome == 1 else 0


    return wins / config.GAMES_PER_WINRATE_ESTIMATE

# NOTE: This is an experimental metric. 
# It uses random ("arbitrary") games to estimate the model's ability to make legal moves. 
# This might be a poor estimate because of how the random games are generated 
# (might be missing endgame/en passent/castling positions), 
# but probably a decent estimate nontheless.
# 
# num_positions = number of positions (chess moves to make) to consider for estimate. 
def estimate_teacher_forced_illegal_probability_mass(model,
                                                    tokenizer,
                                                    arbitrary_game_path,
                                                    num_positions=100):
    mass = 0.0

    # NOTE: rename "arbitrary" to "random"?
    with open(arbitrary_game_path, "r", encoding="utf-8") as f:
        games = [json.loads(line)["moves_uci"] for line in f]

    position_ids = [
        random.randrange(len(games))
        for _ in range(num_positions)
    ]

    with torch.no_grad():
        for game_id in position_ids:
            moves_uci = games[game_id]

            x = [
                tokenizer.BOS_ID
            ] + tokenizer.encode(moves_uci[:-1])

            x = torch.tensor(
                x, 
                dtype=torch.long,
                device=next(model.parameters()).device
                )

            ground_truth_move = torch.tensor(
                tokenizer.encode([moves_uci[-1]]),
                dtype=torch.long,
                device=x.device
                )

            board = construct_board_from_token_ids(
                x.tolist(),
                tokenizer,
            )

            probability_previous = 1.0

            for i in range(3):
                legal_token_ids = get_legal_token_ids(board, x.tolist(), tokenizer) # tolist() expensive perhaps?

                logits = model.forward(x.unsqueeze(0))[0, -1]
                probabilities = torch.softmax(logits, dim=-1)

                illegal_mask = torch.ones_like(
                    probabilities,
                    dtype=torch.bool,
                )
                illegal_mask[list(legal_token_ids)] = False
                illegal_mass = probabilities[illegal_mask].sum()

                mass += (illegal_mass * probability_previous).item()

                probability_previous *= (probabilities[ground_truth_move[i]]).item()

                x = torch.cat(
                    (x, ground_truth_move[i].unsqueeze(0)),
                    dim=0,
                )

    return mass / num_positions
