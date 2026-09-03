import torch
from torch.utils.data import Dataset
import json

from .tokenizer import UCITokenizer

class ChessDataset(Dataset):
    def __init__(self, uci_path, tokenizer):
        self.tokenizer = tokenizer
        self.games = []

        with open(uci_path, "r", encoding="utf-8") as f:
            for line in f:
                game_moves = json.loads(line)['moves_uci']
                encoded_game_moves = self.tokenizer.encode(game_moves)
                self.games.append(encoded_game_moves)

    def __len__(self):
        return len(self.games)

    def __getitem__(self, idx):
        game = self.games[idx]
        inputs = [self.tokenizer.BOS_IDX] + game[:-1]
        targets = game

        return torch.tensor(inputs, dtype=torch.long), torch.tensor(targets, dtype=torch.long)

if __name__ == "__main__":
    # These files are obtained by running scripts/download_data.py
    uci_train_path = "data/subset_100.jsonl"
    uci_val_path = "data/subset_10.jsonl"
    tokenizer = UCITokenizer(uci_train_path)

    train_dataset = ChessDataset(uci_train_path, tokenizer)
    val_dataset = ChessDataset(uci_val_path, tokenizer)

    train_sample = train_dataset.__getitem__(0)

    print("Input moves:")
    print(train_sample[0])
    print("Target moves:")
    print(train_sample[1])