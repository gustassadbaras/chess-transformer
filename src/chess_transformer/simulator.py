import chess
import chess.pgn
import chess.engine
import torch
import random

import config
from .checkpoint import load_checkpoint

def _predict_next_token(x, legal_token_ids, model):
    x_tensor = torch.tensor(
        x,
        dtype=torch.long,
        device=next(model.parameters()).device,
    ).unsqueeze(0)

    logits = model.generate(x_tensor)[0, -1]

    mask = torch.full_like(logits, float("-inf"))
    mask[list(legal_token_ids)] = 0

    probabilities = torch.softmax(logits + mask, dim=-1)

    return torch.multinomial(probabilities, 1).item()

def _move_to_tokens(move):
    uci = move.uci()
    return uci[:2], uci[2:4], uci[4] if len(uci) == 5 else config.SKIP_TOKEN

def construct_board_from_token_ids(token_ids, tokenizer):
    board = chess.Board()
    moves = tokenizer.decode_game(token_ids[1:]) # Skipping <|BOS|>

    for move in moves:
        board.push_uci(move)

    return board

def get_legal_token_ids(board, x, tokenizer):
    legal_moves = [_move_to_tokens(move) for move in board.legal_moves]

    # Subtracting 1 to account for <|BOS|> token
    stage = (len(x) - 1) % 3

    if stage == 0: # source
        legal = {src for src, _, _ in legal_moves}

    elif stage == 1: # target
        src = tokenizer.id2token[x[-1]]
        legal = {
            trg
            for move_src, trg, _ in legal_moves
            if move_src == src
        }

    elif stage == 2: # promotion
        src = tokenizer.id2token[x[-2]]
        trg = tokenizer.id2token[x[-1]]
        legal = {
            promotion
            for move_src, move_trg, promotion in legal_moves
            if move_src == src and move_trg == trg
        }

    return {
        tokenizer.token2id[token]
        for token in legal
    }

# x is a list of split move tokens
def generate_move(board, x, model, tokenizer):
    for _ in range(3):
        legal_ids = get_legal_token_ids(board, x, tokenizer)
        if len(legal_ids) == 0:
            return None
        x.append(_predict_next_token(x, legal_ids, model))

    move_tokens = x[-3:]
    del x[-3:]

    return move_tokens

def make_engine_move(engine, tokenizer, token_id_sequence, board, seconds_per_engine_move):
    result = engine.play(
        board,
        chess.engine.Limit(time=seconds_per_engine_move)
    )
    engine_move = result.move
    engine_move_uci = engine_move.uci()
    token_id_sequence += tokenizer.encode([engine_move_uci])

    board.push(engine_move)


def make_model_move(model, tokenizer, token_id_sequence, board):
    model_move_tokens = generate_move(board, token_id_sequence, model, tokenizer)

    token_id_sequence += model_move_tokens
    model_move_uci = tokenizer.decode_game(model_move_tokens)[0]
    board.push_uci(model_move_uci)

# Returns final board state and -1 if engine won, 0 if draw and 1 if model won
def simulate_game(model, tokenizer, engine_path, elo, seconds_per_engine_move):
    engine = chess.engine.SimpleEngine.popen_uci(engine_path)

    try:
        engine.configure({
            "UCI_LimitStrength": True,
            "UCI_Elo": elo,
        })
        board = chess.Board()
        token_id_sequence = [config.BOS_TOKEN_ID]

        model_plays_first = random.randint(0, 1) == 0
        if model_plays_first:
            make_model_move(model, tokenizer, token_id_sequence, board)
            
        while not board.is_game_over():
            make_engine_move(engine, tokenizer, token_id_sequence, board, seconds_per_engine_move)

            if board.is_game_over():
                break

            make_model_move(model, tokenizer, token_id_sequence, board)
    finally:
        engine.quit()

    if not board.outcome:
        outcome = 0
    elif board.outcome == model_plays_first:
        outcome = 1
    else:
        outcome = -1

    return board, outcome

# For testing
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, tokenizer, optimizer, _ = load_checkpoint(
        r"path-to-checkpoint-pth",
        device)

    ELO = 1320

    board, outcome = simulate_game(model, tokenizer, config.ENGINE_PATH, ELO, config.SECONDS_PER_ENGINE_MOVE)
    print(chess.pgn.Game.from_board(board))