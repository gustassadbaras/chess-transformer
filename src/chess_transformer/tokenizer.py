import config

class UCITokenizer:
    def __init__(self):

        self.UNK_ID = config.UNK_TOKEN_ID
        self.BOS_ID = config.BOS_TOKEN_ID
        self.PAD_ID = config.PAD_TOKEN_ID
        self.SKIP_ID = config.SKIP_TOKEN_ID

        self.token2id = {
            config.UNK_TOKEN : self.UNK_ID,
            config.BOS_TOKEN : self.BOS_ID,
            config.PAD_TOKEN : self.PAD_ID,
            config.SKIP_TOKEN : self.SKIP_ID
            }

        move_vocab = [f"{file}{rank}" for file in "abcdefgh" for rank in range(1,9)] + ['b', 'r', 'n', 'q']
        for move in move_vocab:
            self.token2id[move] = len(self.token2id)

        self.id2token = {v : k for k,v in self.token2id.items()}
        self.vocab_size = len(self.token2id)

    def encode(self, uci_move_list):
        token_ids = []
        for uci_move in uci_move_list:
            token_ids.extend([ # Note: Might want to alert if conversion defaults to <|UNK|>
                self.token2id.get(uci_move[:2], config.UNK_TOKEN_ID),
                self.token2id.get(uci_move[2:4], config.UNK_TOKEN_ID),
                self.token2id.get(uci_move[4], config.UNK_TOKEN_ID) if len(uci_move) == 5 else self.SKIP_ID
            ])

        return token_ids

    def decode(self, id_list):
        return [self.id2token[id] for id in id_list]

    def decode_game(self, id_list):
        uci_move_list = []
        move_tokens = []

        for token_id in id_list:
            if token_id == config.BOS_TOKEN_ID:
                continue

            if token_id in {
                config.PAD_TOKEN_ID,
                config.UNK_TOKEN_ID,
            }:
                break

            move_tokens.append(token_id)

            if len(move_tokens) == 3:
                src_id, trg_id, promotion_id = move_tokens

                src = self.id2token[src_id]
                trg = self.id2token[trg_id]
                promotion = (
                    self.id2token[promotion_id]
                    if promotion_id != config.SKIP_TOKEN_ID
                    else ""
                )

                uci_move_list.append(f"{src}{trg}{promotion}")
                move_tokens = []

        if move_tokens:
            raise ValueError(
                "Token sequence ended in the middle of a move."
            )

        return uci_move_list



if __name__ == "__main__":
    print("Tokenizer test")
    tokenizer = UCITokenizer()

    moves = ['f4f1', 'h4b7', 'f7f8q']

    print(f"Original moves:")
    print(moves)

    print(f"Output of encode():")
    encoded_moves = tokenizer.encode(moves)
    print(encoded_moves)
    
    print(f"Output of decode():")
    print(tokenizer.decode(encoded_moves))

    print(f"Output of decode_game():")
    print(tokenizer.decode_game(encoded_moves))