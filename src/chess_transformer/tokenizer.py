"""
Chess move tokenizer based on UCI protocol. 

# NOTE: current tokenizer is dataset dependent, might want to fix later. Room for experimentation/improvement
"""
from collections import Counter
import json


class UCITokenizer:
    def __init__(self, uci_path):
        counter = Counter()
        with open(uci_path, "r", encoding="utf-8") as f:
            for line in f:
                moves_list = json.loads(line)['moves_uci'] # Assumes json.loads succeeds. Fine perhaps?
                counter.update(moves_list)

        self.UNK_IDX = 0
        self.BOS_IDX = 1
        self.token2idx = {
            "<UNK>" : self.UNK_IDX,
            "<BOS>" : self.BOS_IDX
            }

        for word in sorted(counter):
            self.token2idx[word] = len(self.token2idx)

        self.idx2token = {v : k for k,v in self.token2idx.items()}

    def encode(self, token_list):
        indices = []
        for token in token_list:
            idx = self.token2idx.get(token, self.UNK_IDX)
            indices.append(idx)

        return indices

    def decode(self, idx_list):
        tokens = []
        for idx in idx_list:
            token = self.idx2token.get(idx, self.idx2token[self.UNK_IDX])
            tokens.append(token)

        return tokens


if __name__ == "__main__":
    print("Tokenizer test")
    tokenizer = UCITokenizer("data/subset_100.jsonl")

    moves = ['f4f1', 'f1d3', 'h4h6', '<UNK>']

    print(f"Original moves:")
    print(moves)

    print(f"Encoded moves:")
    encoded_moves = tokenizer.encode(moves)
    print(encoded_moves)
    
    print(f"Decoded moves:")
    print(tokenizer.decode(encoded_moves))