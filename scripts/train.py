import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

# ".src" should be unecessary? Resolve
from src.chess_transformer.experiment_tracker import Experiment
from src.chess_transformer.validation_metrics import estimate_winrate, estimate_teacher_forced_illegal_probability_mass
from src.chess_transformer.dataset import ChessDataset
from src.chess_transformer.dataset import pad_collate_fn
from src.chess_transformer.tokenizer import UCITokenizer
from src.chess_transformer.transformer import Transformer
import config


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    experiment = Experiment(config.EXPERIMENT_TITLE)

    tokenizer = UCITokenizer()

    train_set = ChessDataset(config.UCI_TRAIN_DATA_PATH, tokenizer, config.MAX_SEQUENCE_LENGTH)
    val_set = ChessDataset(config.UCI_VAL_DATA_PATH, tokenizer, config.MAX_SEQUENCE_LENGTH)

    train_loader = DataLoader(train_set, config.BATCH_SIZE, shuffle=True, collate_fn=pad_collate_fn, num_workers=2)
    val_loader = DataLoader(val_set, config.BATCH_SIZE, shuffle=False, collate_fn=pad_collate_fn, num_workers=2)

    model = Transformer(config.NUM_LAYERS, 
                        config.NUM_HEADS, 
                        tokenizer.vocab_size,
                        config.D_MODEL,
                        config.MAX_SEQUENCE_LENGTH).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=config.PAD_TOKEN_ID)
    optimizer = optim.Adam(model.parameters(), 
                           config.LEARNING_RATE) # More configuration would be nice


    global_step = 0
    for epoch in range(1, config.NUM_EPOCHS + 1):
        model.train()
        train_loss = 0
        val_loss = 0
        for batch in train_loader:
            global_step += 1
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)

            predictions = model(inputs)
            loss = criterion(predictions.transpose(1, 2), targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        with torch.no_grad():
            model.eval()
            for batch in val_loader:
                inputs, targets = batch
                inputs = inputs.to(device)
                targets = targets.to(device)

                predictions = model(inputs)
                loss = criterion(predictions.transpose(1, 2), targets)

                val_loss += loss.item()

            average_train_loss = train_loss / len(train_loader)
            average_val_loss = val_loss / len(val_loader)
            winrate = None
            illegal_probability_mass = None

            if (epoch % config.EPOCHS_PER_WINRATE_ESTIMATE == 0 
                and config.ESTIMATE_WINRATE == True):
                experiment.logger.info("Estimating winrate...")
                winrate = estimate_winrate(model, tokenizer,
                                           config.ENGINE_PATH, config.ENGINE_ELO, config.SECONDS_PER_ENGINE_MOVE)
            if (epoch % config.EPOCHS_PER_ILLEGAL_PROBABILITY_MASS_ESTIMATE == 0 
                and config.ESTIMATE_ILLEGAL_PROBABILITY_MASS == True):
                experiment.logger.info("Estimating illegal probability mass...")
                illegal_probability_mass = estimate_teacher_forced_illegal_probability_mass(model,
                                                                                            tokenizer,
                                                                                            config.RANDOM_GAME_PATH)
            if epoch % config.EPOCHS_PER_SAVE == 0:
                experiment.save_checkpoint(model, optimizer, tokenizer,
                                        epoch, global_step, val_loss)
                experiment.logger.info(f"Saved checkpoint @ global step {global_step}, epoch {epoch}")
            if epoch  % config.EPOCHS_PER_GRAD_NORM_LOG == 0:
                experiment.logger.info(f"Logging gradient norms...")
                experiment.log_grad_norms(model, epoch, global_step)

            experiment.log_metrics(epoch, 
                                   global_step,
                                   average_train_loss,
                                   average_val_loss,
                                   winrate if winrate is not None else "NA",
                                   illegal_probability_mass if illegal_probability_mass is not None else "NA")
            experiment.logger.info(f"[EPOCH {epoch} COMPLETE] Train loss: {average_train_loss:.6f}, Val loss: {average_val_loss:.6f}")

if __name__ == "__main__":
    main()