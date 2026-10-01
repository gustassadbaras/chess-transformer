import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import sys

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import os

from src.chess_transformer.checkpoint import load_checkpoint
from src.chess_transformer.experiment_tracker import Experiment
from src.chess_transformer.validation_metrics import estimate_winrate, estimate_teacher_forced_illegal_probability_mass
from src.chess_transformer.dataset import ChessDataset
from src.chess_transformer.dataset import pad_collate_fn
from src.chess_transformer.tokenizer import UCITokenizer
from src.chess_transformer.transformer import Transformer
import config

def setup_distributed():
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1

    if distributed:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")

        dist.init_process_group(
            backend="nccl",
            init_method="env://",
        )
    else:
        rank = 0
        local_rank = 0
        world_size = 1
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    return distributed, rank, local_rank, world_size, device

def cleanup_distributed(distributed):
    if distributed and dist.is_initialized():
        dist.destroy_process_group()

def get_distributed_loss_mean(loss_tracker, device, distributed):
    if not distributed:
        return loss_tracker.mean

    values = torch.tensor(
        [
            loss_tracker.total,
            loss_tracker.count,
        ],
        dtype=torch.float64,
        device=device,
    )

    dist.all_reduce(
        values,
        op=dist.ReduceOp.SUM,
    )

    total_loss = values[0].item()
    total_count = values[1].item()

    return total_loss / total_count


def calculate_val_loss(model, criterion, val_loader, device):
    was_training = model.training
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)

            predictions = model(inputs)
            loss = criterion(predictions.transpose(1, 2), targets)

            val_loss += loss.item()

    if was_training:
        model.train()

    return val_loss / len(val_loader)

def evaluate_and_log(global_step, epoch, 
                 train_loss_mean_estimate, 
                 model, criterion, optimizer, tokenizer, 
                 val_loader, experiment, device):
    val_loss_mean = calculate_val_loss(model, criterion, val_loader, device)
    illegal_probability_mass = None
    winrate = None

    if config.ESTIMATE_WINRATE:
        experiment.logger.info("Estimating winrate...")
        winrate = estimate_winrate(model, tokenizer,
                                config.ENGINE_PATH, config.ENGINE_ELO, config.SECONDS_PER_ENGINE_MOVE)
    if config.ESTIMATE_ILLEGAL_PROBABILITY_MASS:
        experiment.logger.info("Estimating illegal probability mass...")
        illegal_probability_mass = estimate_teacher_forced_illegal_probability_mass(model,
                                                                                    tokenizer,
                                                                                    config.RANDOM_GAME_PATH)
    experiment.save_checkpoint(model, optimizer, tokenizer,
                            epoch, global_step, val_loss_mean)
    experiment.logger.info(f"Saved checkpoint @ global step {global_step}, epoch {epoch}")
    experiment.log_grad_norms(model, epoch, global_step)

    experiment.log_metrics(epoch, 
                        global_step,
                        train_loss_mean_estimate,
                        val_loss_mean,
                        winrate if winrate is not None else "NA",
                        illegal_probability_mass if illegal_probability_mass is not None else "NA")
    experiment.logger.info(f"[GLOBAL STEP {global_step}, EPOCH {epoch}] Train loss: {train_loss_mean_estimate:.6f}, Val loss: {val_loss_mean:.6f}")

    return

class LossTracker:
    def __init__(self):
        self.total = 0.0
        self.count = 0

    def update(self, loss, count):
        self.total += loss * count
        self.count += count

    @property 
    def mean(self):
        return self.total / self.count

    def reset(self):
        self.total = 0.0
        self.count = 0

def train(checkpoint_path=None):
    (distributed, rank, local_rank, world_size, device) = setup_distributed()

    is_main_process = rank == 0

    if is_main_process:
        print(
            f"Using {world_size} GPU(s)"
            if distributed
            else f"Using device: {device}"
        )
    
    if is_main_process:
        experiment = Experiment(
            config.EXPERIMENT_TITLE
        )
    

    if checkpoint_path is None:
        tokenizer = UCITokenizer()
        model = Transformer(config.NUM_LAYERS, 
                            config.NUM_HEADS, 
                            tokenizer.vocab_size,
                            config.D_MODEL,
                            config.MAX_SEQUENCE_LENGTH).to(device)

        optimizer = optim.Adam(model.parameters(), 
                            config.LEARNING_RATE) # More configuration would be nice
        global_step = 0
        start_epoch = 1
    else:
        model, tokenizer, optimizer, checkpoint_epoch, global_step, val_loss = (
            load_checkpoint(checkpoint_path,
                            device)
        )
        print(f"Resuming training @ global step {global_step}, "
              f"epoch {checkpoint_epoch}, with validation loss {val_loss}")
        start_epoch = checkpoint_epoch + 1

    if distributed:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
        )

    criterion = nn.CrossEntropyLoss(ignore_index=config.PAD_TOKEN_ID)

    train_set = ChessDataset(config.UCI_TRAIN_DATA_PATH, tokenizer, config.MAX_SEQUENCE_LENGTH)

    if distributed:
        train_sampler = DistributedSampler(
            train_set,
            num_replicas=world_size,
            rank=rank,
            shuffle=True)
    else:
        train_sampler = None

    train_loader = DataLoader(
        train_set,
        batch_size=config.BATCH_SIZE,
        shuffle=(train_sampler is None), 
        sampler=train_sampler,
        collate_fn=pad_collate_fn,
        num_workers=2,
        pin_memory=device.type == "cuda"
    )

    if is_main_process:
        val_set = ChessDataset(config.UCI_VAL_DATA_PATH, tokenizer, config.MAX_SEQUENCE_LENGTH)
        val_loader = DataLoader(
            val_set,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            collate_fn=pad_collate_fn,
            num_workers=2,
            pin_memory=device.type == "cuda"
        )

    train_loss_tracker = LossTracker()
    for epoch in range(start_epoch, config.NUM_EPOCHS + 1):
        model.train()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
            
        for batch in train_loader:
            global_step += 1
            inputs, targets = batch
            inputs = inputs.to(
                device,
                non_blocking=True)
            targets = targets.to(
                device,
                non_blocking=True)

            optimizer.zero_grad()
            predictions = model(inputs)
            loss = criterion(predictions.transpose(1, 2), 
                             targets)

            loss.backward()
            optimizer.step()

            train_loss_tracker.update(loss.item(), 
                                      targets.shape[0])          

            should_log = (
                global_step % config.GLOBAL_STEPS_PER_METRIC_LOG == 0
            )
            if should_log:
                train_loss_mean = get_distributed_loss_mean(
                    train_loss_tracker,
                    device,
                    distributed)

                if is_main_process:
                    model_for_logging = (
                        model.module
                        if distributed
                        else model
                        )

                    if distributed:
                        dist.barrier()

                    # Too many args, would be nice to clean up at some point
                    evaluate_and_log(global_step, epoch, train_loss_mean, 
                                     model_for_logging, criterion, optimizer, 
                                     tokenizer, val_loader, experiment, device)

                    if distributed:
                        dist.barrier()
                else:
                    if distributed:
                        dist.barrier()
                        dist.barrier()

                train_loss_tracker.reset()

    cleanup_distributed(distributed)

if __name__ == "__main__":
    if len(sys.argv) == 1:
        checkpoint_path = None
    elif len(sys.argv) == 3 and sys.argv[1] == "--resume":
        checkpoint_path = sys.argv[2]
    else:
        raise ValueError("Usage: python train.py [--resume <checkpoint>]")

    train(checkpoint_path)