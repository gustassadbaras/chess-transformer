import torch
import random
import chess
import random
import math

from src.chess_transformer.checkpoint import load_checkpoint
from src.chess_transformer.simulator import make_model_move
import config

MODEL_1_PATH = r"models\25112.pth"
MODEL_2_PATH = r"models\20732.pth"

def play_game(model_1, model_2, tokenizer):

    board = chess.Board()

    model_1_plays_first = random.randint(0, 1) == 0
    if model_1_plays_first:
        make_model_move(model_1, tokenizer, board)
        
    while not board.is_game_over():
        make_model_move(model_2, tokenizer, board)

        if board.is_game_over():
            break

        make_model_move(model_1, tokenizer, board)


    outcome = board.outcome()

    if outcome is None or outcome.winner is None:
        result = 0  # Draw
    else:
        model_1_color = chess.WHITE if model_1_plays_first else chess.BLACK
        result = 1 if outcome.winner == model_1_color else -1

    print(f"Model 1 result: {'Win' if result == 1 else 'Loss' if result == -1 else 'Draw'}")

    return result


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_1, tokenizer, optimizer, checkpoint_epoch, global_step, val_loss = (
        load_checkpoint(MODEL_1_PATH,
        device))
    trainable_params = sum(p.numel() for p in model_1.parameters() if p.requires_grad)
    print(f"Param count: {trainable_params:,}")

    model_2, tokenizer, optimizer, checkpoint_epoch, global_step, val_loss = (
        load_checkpoint(MODEL_2_PATH,
        device))

    wins = 0
    losses = 0
    draws = 0
    for i in range(100):
        result = play_game(model_1, model_2, tokenizer)

        if result == 1:
            wins += 1
        elif result == -1:
            losses += 1
        else:
            draws += 1

        print(f"wins: {wins}, losses: {losses}, draws: {draws}")

    total_games = wins + losses + draws

    # Elo estimate: win = 1 point, draw = 0.5 points, loss = 0 points
    score = wins + 0.5 * draws
    p = score / total_games

    if p == 1.0:
        print("Estimated Elo difference: Model 1 is infinitely stronger (no losses observed).")
    elif p == 0.0:
        print("Estimated Elo difference: Model 1 is infinitely weaker (no points scored).")
    else:
        elo_diff = 400 * math.log10(p / (1 - p))
        print(f"Estimated Elo difference (Model 1 - Model 2): {elo_diff:.1f}")