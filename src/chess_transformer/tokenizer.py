import chess
import config

class UCITokenizer:
    def __init__(self):
        self.input_token2id = {
            ' ' : 0,
            'P': 1,
            'N': 2,
            'B': 3,
            'R': 4,
            'Q': 5,
            'K': 6,

            'p': 7,
            'n': 8,
            'b': 9,
            'r': 10,
            'q': 11,
            'k': 12,

            'EP': 13, # white pawn which just pushed 2 squares
            'Ep': 14, # black pawn which just pushed 2 squares
            'wtm': 15, # white to move
            'btm': 16, # black to move
            'w_kingside_castle_false' : 17,
            'w_kingside_castle_true' : 18,
            'w_queenside_castle_false' : 19,
            'w_queenside_castle_true' : 20,
            'b_kingside_castle_false' : 21,
            'b_kingside_castle_true' : 22,
            'b_queenside_castle_false' : 23,
            'b_queenside_castle_true' : 24,
            'CLS' : 25
        }
        self.input_vocab_size = len(self.input_token2id)

        self.output_token2id = generate_uci_move_vocab()

        self.output_id2token = {v : k for k,v in self.output_token2id.items()}
        self.output_vocab_size = len(self.output_token2id)

    # NOTE: this function is misleading. "uci_move_list" as a parm, but this doesn't operate on lists of strings. 
    def encode(self, uci_move_list):
        token_ids = []
        for uci_move in uci_move_list:
            token_ids.append(self.output_token2id[uci_move])

        return token_ids

    def decode(self, id_list):
        return [self.output_id2token[id] for id in id_list]


def generate_uci_move_vocab():
    files = "abcdefgh"
    ranks = "12345678"

    def sq_name(f, r):
        return f"{files[f]}{ranks[r]}"

    queen_dirs = [(1, 0), (-1, 0), (0, 1), (0, -1),
                  (1, 1), (1, -1), (-1, 1), (-1, -1)]
    knight_deltas = [(1, 2), (2, 1), (-1, 2), (-2, 1),
                      (1, -2), (2, -1), (-1, -2), (-2, -1)]

    moves = []

    for f in range(8):
        for r in range(8):
            from_sq = sq_name(f, r)

            for df, dr in queen_dirs:
                for dist in range(1, 8):
                    tf, tr = f + df * dist, r + dr * dist
                    if 0 <= tf < 8 and 0 <= tr < 8:
                        moves.append(from_sq + sq_name(tf, tr))
                    else:
                        break

            for df, dr in knight_deltas:
                tf, tr = f + df, r + dr
                if 0 <= tf < 8 and 0 <= tr < 8:
                    moves.append(from_sq + sq_name(tf, tr))

    promo_pieces = "qnbr"
    for f in range(8):
        for (start_r, end_r) in [(6, 7), (1, 0)]:  # rank7->8, rank2->1 (0-indexed)
            for df in (-1, 0, 1):
                tf = f + df
                if 0 <= tf < 8:
                    base = sq_name(f, start_r) + sq_name(tf, end_r)
                    for piece in promo_pieces:
                        moves.append(base + piece)

    # Deduplicate while preserving order, just in case
    seen = set()
    unique_moves = []
    for m in moves:
        if m not in seen:
            seen.add(m)
            unique_moves.append(m)

    return {move: idx for idx, move in enumerate(unique_moves)}

if __name__ == "__main__":
    print("Tokenizer test")
    tokenizer = UCITokenizer()

    moves = ['f4f1', 'g8h6', 'f7f8q']

    print(f"Original moves:")
    print(moves)

    print(f"Output of encode():")
    encoded_moves = tokenizer.encode(moves)
    print(encoded_moves)
    
    print(f"Output of decode():")
    print(tokenizer.decode(encoded_moves))

    print(f"Output of decode_game():")
    print(tokenizer.decode_game(encoded_moves))