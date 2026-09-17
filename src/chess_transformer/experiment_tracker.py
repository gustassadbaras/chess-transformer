from pathlib import Path
from datetime import datetime
import shutil
import csv
import logging

from .checkpoint import save_checkpoint

class Experiment():
    def __init__(self, experiment_title):
        start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.experiment_dir_path = Path("experiments") / f"{experiment_title}_{start_time}"
        self.log_file_path = self.experiment_dir_path / "train.log"
        self.metrics_file_path = self.experiment_dir_path / "metrics.csv"
        self.grad_norm_file_path = self.experiment_dir_path / "grad_norms.csv"
        self.checkpoint_dir_path = self.experiment_dir_path / "checkpoints"
        self.config_save_path = self.experiment_dir_path / "config.py"

        # NOTE: if two experiments are somehow run on the same second, this would fail. 
        self.experiment_dir_path.mkdir()
        self.checkpoint_dir_path.mkdir()

        self.logger = self.setup_logger(self.log_file_path)

        shutil.copy("config.py", self.config_save_path)

    def setup_logger(self, log_file_path):
        logger = logging.getLogger("chess_transformer")
        logger.setLevel(logging.INFO)

        formatter = logging.Formatter(
            "[%(asctime)s] - %(message)s"
        )

        file_handler = logging.FileHandler(log_file_path)
        file_handler.setFormatter(formatter)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        logger.handlers.clear()
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger

    # Shouldn't this be in logger.py?
    def log_metrics(self, epoch, global_step, 
                    train_loss, val_loss,
                    winrate, illegal_probability_mass):
        file_exists = self.metrics_file_path.exists()

        with self.metrics_file_path.open("a", newline="") as f:
            writer = csv.writer(f)

            if not file_exists:
                writer.writerow([
                    "epoch",
                    "global_step",
                    "train_loss",
                    "val_loss",
                    "winrate",
                    "illegal_probability_mass"
                ])

            writer.writerow([
                epoch,
                global_step,
                f"{train_loss:.6f}",
                f"{val_loss:.6f}",
                f"{winrate}",
                f"{illegal_probability_mass}",
            ])

    def log_grad_norms(self, model, epoch, global_step):
        # getting gradient norms
        grad_norms = {}

        for name, module in model.named_modules():
            parameters = [
                p for p in module.parameters(recurse=False)
                if p.grad is not None
            ]

            if not parameters:
                continue

            squared_norm = sum(
                p.grad.detach().pow(2).sum().item()
                for p in parameters
            )

            grad_norms[name] = squared_norm ** 0.5

        # logging to file
        file_exists = self.grad_norm_file_path.exists()

        with self.grad_norm_file_path.open("a", newline="") as f:
            writer = csv.writer(f)

            if not file_exists:
                writer.writerow(
                    ["epoch", "global_step"] +
                    list(grad_norms.keys())
                    )

            writer.writerow(
                [epoch, global_step] +
                list(grad_norms.values())
                )

    def save_checkpoint(self, 
                        model, 
                        optimizer, 
                        tokenizer,
                        epoch,
                        global_step,
                        val_loss
                        ):
        path = self.checkpoint_dir_path / f"{global_step}.pth"
        save_checkpoint(model, 
                        optimizer, 
                        tokenizer, 
                        epoch,
                        global_step,
                        val_loss, # omitting training loss because it doesn't seem useful here, val loss might be important for picking tho?
                        path)
