import chess
import chess.pgn
import chess.engine
import torch
import random
import numpy as np

import config
from .checkpoint import load_checkpoint

def _predict_next_token(x, legal_token_ids, model):
    x_tensor = torch.tensor(
        x,
        dtype=torch.long,
        device=next(model.parameters()).device,
    ).unsqueeze(0)

    logits = model.forward(x_tensor)[0]

    mask = torch.full_like(logits, float("-inf"))
    mask[list(legal_token_ids)] = 0

    probabilities = torch.softmax(logits + mask, dim=-1)

    return torch.multinomial(probabilities, 1).item()

def construct_board_from_token_ids(token_ids, tokenizer):
    board = chess.Board()
    moves = tokenizer.decode(token_ids)

    for move in moves:
        board.push_uci(move)

    return board

def get_legal_token_ids(board, tokenizer):
    legal_moves = [move.uci() for move in board.legal_moves]

    return tokenizer.encode(legal_moves)


def encode_fen(fen):
    FEN_PIECE = {c: i + 1 for i, c in enumerate("PNBRQK")}
    FEN_PIECE |= {c: i + 7 for i, c in enumerate("pnbrqk")}
    board_str, turn, castling, ep, *_ = fen.split()
    tok = np.zeros(config.BOARD_SEQUENCE_LENGTH, dtype=np.uint8)

    # Board: FEN lists rank 8 first, our index is rank*8 + file with a1 = 0.
    for r, row in enumerate(board_str.split('/')):
        rank = 7 - r
        file = 0
        for ch in row:
            if ch.isdigit():
                file += int(ch)
            else:
                tok[rank * 8 + file] = FEN_PIECE[ch]
                file += 1

    if ep != '-':
        ep_sq = (ord(ep[0]) - 97) + 8 * (int(ep[1]) - 1)
        if ep[1] == '3':          # white just pushed, pawn is one rank above
            tok[ep_sq + 8] = 13
        else:                     # '6': black just pushed, pawn is one rank below
            tok[ep_sq - 8] = 14

    tok[64] = 15 if turn == 'w' else 16

    for k, c in enumerate("KQkq"):   # '-' simply matches nothing -> all "no right"
        tok[65 + k] = 17 + 2 * k + (c in castling)

    tok[-1] = config.CLS_ID
    return tok

def generate_move(board, model, tokenizer):
    legal_ids = get_legal_token_ids(board, tokenizer)
    if len(legal_ids) == 0:
        return None

    x = encode_fen(board.fen())
    
    return _predict_next_token(x, legal_ids, model)

def make_engine_move(engine, board, seconds_per_engine_move):
    result = engine.play(
        board,
        chess.engine.Limit(time=seconds_per_engine_move)
    )
    engine_move = result.move

    board.push(engine_move)


def make_model_move(model, tokenizer, board):
    model_move_token = generate_move(board, model, tokenizer)

    model_move_uci = tokenizer.decode([model_move_token])[0]
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

        model_plays_first = random.randint(0, 1) == 0
        if model_plays_first:
            make_model_move(model, tokenizer, board)
            
        while not board.is_game_over():
            make_engine_move(engine, board, seconds_per_engine_move)

            if board.is_game_over():
                break

            make_model_move(model, tokenizer, board)
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

    model, tokenizer, optimizer, _, _, _ = load_checkpoint(
        r"path-to-checkpoint-pth",
        device)

    ELO = 1320

    board, outcome = simulate_game(model, tokenizer, config.ENGINE_PATH, ELO, config.SECONDS_PER_ENGINE_MOVE)
    print(chess.pgn.Game.from_board(board))