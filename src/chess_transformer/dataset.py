import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pyarrow.parquet as pq
from numba import njit, prange
import numpy as np
import math
import os

import config


# ---- token layout (69 tokens) ------------------------------------------
# 0..63  squares, index = rank*8 + file (a1=0, h1=7, a8=56)
# 64     turn
# 65..68 castling slots K, Q, k, q
# square tokens: 0 empty, 1-6 white P N B R Q K, 7-12 black P N B R Q K,
#                13 white pawn that just double-pushed (EP-able), 14 same for black
# turn tokens: 15 white to move, 16 black to move
# castling: 17 + 2*slot + (0 = no right, 1 = right)  -> ids 17..24

# NOTE: move elsewhere later
START_BOARD = np.zeros(64, dtype=np.int8)
START_BOARD[0:8] = [4, 2, 3, 5, 6, 3, 2, 4]
START_BOARD[8:16] = 1
START_BOARD[48:56] = 7
START_BOARD[56:64] = [10, 8, 9, 11, 12, 9, 8, 10]

PROMO = {'n': 2, 'b': 3, 'r': 4, 'q': 5}


def build_move_tables(token2id, vocab_size):
    mfrom = np.full(vocab_size, -1, dtype=np.int16)
    mto = np.full(vocab_size, -1, dtype=np.int16)
    mpromo = np.zeros(vocab_size, dtype=np.int8)
    for tok, tid in token2id.items():
        if len(tok) not in (4, 5) or tok[0] not in 'abcdefgh' or tok[1] not in '12345678':
            continue
        mfrom[tid] = (ord(tok[0]) - 97) + 8 * (int(tok[1]) - 1)
        mto[tid] = (ord(tok[2]) - 97) + 8 * (int(tok[3]) - 1)
        if len(tok) == 5:
            mpromo[tid] = PROMO[tok[4]]
    return mfrom, mto, mpromo


@njit(parallel=True, cache=True)
def fill_positions(idx, flat, offsets, mfrom, mto, 
                   mpromo, start_board, out_tok, out_tgt):
    for b in prange(idx.shape[0]):
        i = idx[b]
        g = np.searchsorted(offsets, i, side='right') - 1
        ply = i - offsets[g]
        base = offsets[g]

        board = start_board.copy()
        castle = np.ones(4, dtype=np.int8)  # K, Q, k, q
        ep = -1

        for k in range(ply):
            m = flat[base + k]
            f = mfrom[m]
            t = mto[m]
            piece = board[f]
            is_black = piece >= 7
            ptype = piece - 6 if is_black else piece

            # en passant capture: pawn moves diagonally onto an empty square
            if ptype == 1 and (f % 8) != (t % 8) and board[t] == 0:
                board[t + 8 if is_black else t - 8] = 0

            # castling: king moves two files, move the rook too
            if ptype == 6 and abs(t - f) == 2:
                if t > f:
                    board[f + 1] = board[f + 3]
                    board[f + 3] = 0
                else:
                    board[f - 1] = board[f - 4]
                    board[f - 4] = 0

            promo = mpromo[m]
            if promo > 0:
                board[t] = promo + 6 if is_black else promo
            else:
                board[t] = piece
            board[f] = 0

            # en passant flag: set after every double push
            if ptype == 1 and abs(t - f) == 16:
                ep = t
            else:
                ep = -1

            # castling rights: any move touching a king/rook home square
            if f == 4 or t == 4:
                castle[0] = 0; castle[1] = 0
            if f == 60 or t == 60:
                castle[2] = 0; castle[3] = 0
            if f == 7 or t == 7:
                castle[0] = 0
            if f == 0 or t == 0:
                castle[1] = 0
            if f == 63 or t == 63:
                castle[2] = 0
            if f == 56 or t == 56:
                castle[3] = 0

        for s in range(64):
            out_tok[b, s] = board[s]
        if ep >= 0:
            out_tok[b, ep] = 13 if board[ep] == 1 else 14
        out_tok[b, 64] = 15 + (ply & 1)
        for k in range(4):
            out_tok[b, 65 + k] = 17 + 2 * k + castle[k]
        out_tgt[b] = flat[base + ply]

def encode_board(board, ds):
    moves = [ds.tokenizer.token2id[m.uci()] for m in board.move_stack]
    n = len(moves)

    # One game: [m1..mn, dummy]. The dummy is only read as the (unused) target.
    flat = np.array(moves + [0], dtype=np.int32)
    offsets = np.array([0, n + 1], dtype=np.int64)
    idx = np.array([n], dtype=np.int64)

    tok = np.empty((1, config.BOARD_SEQUENCE_LENGTH), dtype=np.uint8)
    tgt = np.empty(1, dtype=np.int64)
    fill_positions(idx, flat, offsets,
                   ds.mfrom, ds.mto, ds.mpromo, START_BOARD, tok, tgt)
    tok[:, -1] = config.CLS_ID
    return tok[0]

class ChessDataset(Dataset):
    def __init__(self, uci_path, tokenizer, max_sequence_length):
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length
        self.vocab_size = len(tokenizer.output_token2id)

        dataset_name = uci_path.split('/')[-1].split('.')[0]
        flat_save_path = f"data/cached_flat_{dataset_name}.npy"
        offset_save_path = f"data/cached_offset_{dataset_name}.npy"
        if os.path.exists(flat_save_path) and os.path.exists(offset_save_path):
            print(f"Loading cached dataset {dataset_name} ({uci_path})")
            self.flat_tokens = np.load(flat_save_path)
            self.offsets = np.load(offset_save_path)
        else:
            print(f"Loading dataset {dataset_name} for the first time ({uci_path})")
            game_chunks = []
            game_lengths = []

            parquet_file = pq.ParquetFile(uci_path)
            metadata = pq.read_metadata(uci_path)
            num_games = metadata.num_rows
            total_batches = math.ceil(num_games / 10000)
            for batch_idx, batch in enumerate(
                parquet_file.iter_batches(columns=["moves_uci"], batch_size=10000)):
                for game_moves in batch.column("moves_uci").to_pylist():
                    del game_moves[max_sequence_length:]
                    game_tokens = tokenizer.encode(game_moves)[:max_sequence_length]
                    game_chunks.append(np.asarray(game_tokens, dtype=np.int32))
                    game_lengths.append(len(game_tokens))
                print(
                    f"\rLoading dataset consisting of {num_games} chess games. "
                    f"Progress: {batch_idx+1}/{total_batches} "
                    f"({(batch_idx+1) / total_batches:.1%})",
                    end="",
                    flush=True,
                )

            self.flat_tokens = np.concatenate(game_chunks)
            self.offsets = np.zeros(len(game_lengths) + 1, dtype=np.int64)
            np.cumsum(game_lengths, out=self.offsets[1:])

            np.save(flat_save_path, self.flat_tokens)
            np.save(offset_save_path, self.offsets)

        self.flat_tokens = np.ascontiguousarray(self.flat_tokens, dtype=np.int32)
        self.offsets = np.ascontiguousarray(self.offsets, dtype=np.int64)
        self.num_positions = int(self.offsets[-1])

        self.mfrom, self.mto, self.mpromo = build_move_tables(
            tokenizer.output_token2id, len(tokenizer.output_token2id))

        print("\nSuccessfully loaded dataset.")

    def __len__(self):
        return self.num_positions

    def get_batch(self, idx):
        idx = np.ascontiguousarray(idx, dtype=np.int64)
        tok = np.empty((len(idx), config.BOARD_SEQUENCE_LENGTH), dtype=np.uint8)
        tgt = np.empty(len(idx), dtype=np.int64)
        fill_positions(idx, self.flat_tokens, self.offsets, self.mfrom, 
                       self.mto, self.mpromo, START_BOARD, tok, tgt)
        tok[:, -1] = config.CLS_ID
        return tok, tgt

    def __getitem__(self, i):
        tok, tgt = self.get_batch(np.array([i]))
        return torch.from_numpy(tok[0]).long(), torch.tensor(tgt[0])


if __name__ == "__main__":
    import chess
    from .tokenizer import UCITokenizer

    tokenizer = UCITokenizer()

    ds = ChessDataset("data/val_set.parquet",
                      tokenizer,
                      256)
    
    for g in range(10000):
    # for g in range(len(ds.offsets)-1):
        moves = ds.flat_tokens[ds.offsets[g]: ds.offsets[g + 1]]
        uci = [ds.tokenizer.output_id2token[m] for m in moves]
        board = chess.Board()
        idxs = np.arange(ds.offsets[g], ds.offsets[g + 1])
        tok, _ = ds.get_batch(idxs)
        for ply, u in enumerate(uci):
            for sq in range(64):
                p = board.piece_at(sq)
                expect = 0 if p is None else p.piece_type + (6 if not p.color else 0)
                got = tok[ply, sq]
                if got in (13, 14): got = 1 if got == 13 else 7
                assert got == expect, (g, ply, sq)
            assert tok[ply, 64] == 15 + (ply & 1)
            rights = [board.has_kingside_castling_rights(True), board.has_queenside_castling_rights(True),
                    board.has_kingside_castling_rights(False), board.has_queenside_castling_rights(False)]
            assert all(tok[ply, 65 + k] == 17 + 2 * k + int(r) for k, r in enumerate(rights)), (g, ply)
            board.push_uci(u)

    print("Encoding working successfully")