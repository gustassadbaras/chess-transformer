import torch
import torch.nn as nn
from torch.utils.data import BatchSampler, DataLoader
import torch.optim as optim

import argparse
from pathlib import Path

from src.chess_transformer.stockfish_dataset import StockfishDataset
from src.chess_transformer.checkpoint import load_checkpoint
from src.chess_transformer.experiment_tracker import Experiment
from .train import LossTracker # should make a module with utility funcitons for this, not keep it in train.py

def parse_args():
    parser = argparse.ArgumentParser(description="DAgger training")

    parser.add_argument(
        "--model-path", type=Path,
        default=Path("models/model.pth")
    )
    parser.add_argument(
        "--original-dataset-path", type=Path,
        default=Path("data/relabeled/combined_train.parquet")
    )
    parser.add_argument(
        "--dagger-dataset-path", type=Path,
        default=Path("data/dagger/dagger.parquet")
    )

    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--dagger-fraction", type=float, default=0.3)
    parser.add_argument(
        "--global-steps-per-checkpoint", type=int, default=100
    )

    parser.add_argument(
        "--dagger-train-split-fraction", type=float, default=0.8
    )
    parser.add_argument(
        "--original-train-split-fraction", type=float, default=0.9
    )

    return parser.parse_args()


args = parse_args()

MODEL_PATH = args.model_path
ORIGINAL_DATASET_PATH = args.original_dataset_path
DAGGER_DATASET_PATH = args.dagger_dataset_path

NUM_EPOCHS = args.num_epochs
GLOBAL_STEPS_PER_CHECKPOINT = args.global_steps_per_checkpoint
LEARNING_RATE = args.learning_rate
BATCH_SIZE = args.batch_size

DAGGER_FRACTION = args.dagger_fraction

DAGGER_TRAIN_VAL_SPLIT = [
    args.dagger_train_split_fraction,
    1 - args.dagger_train_split_fraction
]

ORIGINAL_TRAIN_VAL_SPLIT = [
    args.original_train_split_fraction,
    1 - args.original_train_split_fraction
]

class MixedBatchSampler(BatchSampler):
    def __init__(self, original_dataset, dagger_dataset, dagger_fraction, batch_size):
        self.original_dataset = original_dataset
        self.dagger_dataset = dagger_dataset
        self.batch_size = batch_size
        self.dagger_count = int(dagger_fraction * batch_size)
        self.original_count = batch_size - self.dagger_count

    def __iter__(self):
        dagger_indices = torch.randperm(len(self.dagger_dataset))
        original_indices = torch.randperm(len(self.original_dataset))


        for i in range(min(len(dagger_indices) // self.dagger_count, 
                           len(original_indices) // self.original_count)):
            batch = (
                dagger_indices[i * self.dagger_count:(i + 1) * self.dagger_count].tolist()
                    + (original_indices[i * self.original_count:(i + 1) * self.original_count]
                    + len(self.dagger_dataset)
                ).tolist()
            )
            yield batch

    def __len__(self):
        return min(
            len(self.dagger_dataset) // self.dagger_count,
            len(self.original_dataset) // self.original_count
        )

def calculate_validation_loss(model, criterion, loader, device):
    was_training = model.training
    model.eval()

    total_loss = 0
    total_samples = 0
    for batch in loader:
        inputs, targets = batch
        inputs = inputs.to(device)
        targets = targets.to(device)

        predictions = model(inputs)
        total_loss += criterion(predictions, targets).item() * targets.size(0)
        total_samples += targets.size(0)

    model.train(was_training)
    return total_loss / total_samples

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu")

experiment = Experiment("DAgger")

model, tokenizer, *_ = (
    load_checkpoint(MODEL_PATH,
                    device))

original_dataset = StockfishDataset(ORIGINAL_DATASET_PATH, tokenizer)
dagger_dataset = StockfishDataset(DAGGER_DATASET_PATH, tokenizer)

from torch.utils.data import (
    DataLoader, ConcatDataset, random_split
)

dagger_train, dagger_val = random_split(
    dagger_dataset, DAGGER_TRAIN_VAL_SPLIT
)
original_train, original_val = random_split(
    original_dataset, ORIGINAL_TRAIN_VAL_SPLIT
)

train_dataset = ConcatDataset([dagger_train, original_train])

train_loader = DataLoader(
    train_dataset,
    batch_sampler=MixedBatchSampler(
        original_train,
        dagger_train,
        dagger_fraction=DAGGER_FRACTION,
        batch_size=BATCH_SIZE,
    )
)

dagger_val_loader = DataLoader(dagger_val, batch_size=BATCH_SIZE)
original_val_loader = DataLoader(original_val, batch_size=BATCH_SIZE)

optimizer = optim.Adam(model.parameters(), 
                       lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

train_loss_tracker = LossTracker()
global_step = 0
for epoch in range(NUM_EPOCHS):
    model.train()
        
    for batch in train_loader:
        print("batch")
        global_step += 1
        inputs, targets = batch
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        predictions = model(inputs)
        loss = criterion(predictions, targets)

        loss.backward()
        optimizer.step()

        train_loss_tracker.update(loss.item(), 
                                    targets.shape[0])          


        if global_step % GLOBAL_STEPS_PER_CHECKPOINT == 0:
            with torch.no_grad():
                dagger_val_loss = calculate_validation_loss(model, criterion, dagger_val_loader, device)
                original_val_loss = calculate_validation_loss(model, criterion, original_val_loader, device)
                print(f"[EPOCH {epoch}]",
                    f"\n\tTrain loss: {train_loss_tracker.mean}",
                    f"\n\tDAgger val loss: {dagger_val_loss}",
                    f"\n\tOriginal val loss: {original_val_loss}")

                experiment.save_checkpoint(model, optimizer, tokenizer,
                            epoch, global_step, dagger_val_loader)