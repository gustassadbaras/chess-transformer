import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import json

import config

class ChessDataset(Dataset):
    def __init__(self, uci_path, tokenizer, max_sequence_length):
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length
        self.vocab_size = len(tokenizer.token2id)
        self.games = []

        with open(uci_path, "r", encoding="utf-8") as f:
            for line in f:
                game_moves = json.loads(line)['moves_uci']
                # Might be inefficient to extend list like this
                encoded_game_moves = [config.BOS_TOKEN_ID] + self.tokenizer.encode(game_moves)
                del encoded_game_moves[max_sequence_length:]
                self.games.append(encoded_game_moves)

    def __len__(self):
        return len(self.games)

    def __getitem__(self, id):
        game = self.games[id]
        inputs = game[:-1]
        targets = game[1:]
        # perhaps this would be better:
        # del game[0]
        # targets = game
        # ?

        return torch.tensor(inputs, dtype=torch.long), torch.tensor(targets, dtype=torch.long)

def pad_collate_fn(batch):
    inputs, targets = zip(*batch)

    padded_inputs = pad_sequence(
        inputs,
        batch_first=True,
        padding_value=config.PAD_TOKEN_ID
    )

    padded_targets = pad_sequence(
        targets,
        batch_first=True,
        padding_value=config.PAD_TOKEN_ID
    )
    
    return padded_inputs, padded_targets