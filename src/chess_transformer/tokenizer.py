import chess
import config

class UCITokenizer:
    def __init__(self):
        self.UNK_ID = config.UNK_TOKEN_ID
        self.BOS_ID = config.BOS_TOKEN_ID
        self.PAD_ID = config.PAD_TOKEN_ID

        self.token2id = {
            config.UNK_TOKEN : self.UNK_ID,
            config.BOS_TOKEN : self.BOS_ID,
            config.PAD_TOKEN : self.PAD_ID
            }

        self.token2id.update(generate_uci_move_vocab())

        self.id2token = {v : k for k,v in self.token2id.items()}
        self.vocab_size = len(self.token2id)

    def encode(self, uci_move_list):
        token_ids = []
        for uci_move in uci_move_list:
            token_ids.append(self.token2id[uci_move])

        return token_ids

    def decode(self, id_list):
        return [self.id2token[id] for id in id_list]

    def decode_game(self, id_list):
        uci_move_list = []

        for token_id in id_list:
            if token_id == config.BOS_TOKEN_ID:
                continue

            if token_id in {
                config.PAD_TOKEN_ID,
                config.UNK_TOKEN_ID,
            }:
                break

            uci_move_list.append(self.id2token[token_id])

        return uci_move_list

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

    moves = ['<|BOS|>', 'f4f1', 'g8h6', 'f7f8q']

    print(f"Original moves:")
    print(moves)

    print(f"Output of encode():")
    encoded_moves = tokenizer.encode(moves)
    print(encoded_moves)
    
    print(f"Output of decode():")
    print(tokenizer.decode(encoded_moves))

    print(f"Output of decode_game():")
    print(tokenizer.decode_game(encoded_moves))