import torch
import config

from .transformer import Transformer

def save_checkpoint(model, 
                    optimizer, 
                    tokenizer, 
                    epoch,
                    global_step,
                    val_loss,
                    path):
    torch.save({"epoch" : epoch,
                "global_step" : global_step,
                "val_loss" : val_loss,
                "model_state_dict" : model.state_dict(),
                "optimizer_state_dict" : optimizer.state_dict(), # Assuming Adam is used 
                "model_params": {
                    "num_layers" : config.NUM_LAYERS,
                    "num_heads" : config.NUM_HEADS,
                    "vocab_size" : model.vocab_size,
                    "d_model" : config.D_MODEL,
                    "max_sequence_length" : config.MAX_SEQUENCE_LENGTH
                    },
                "tokenizer" : tokenizer,
                },
                path)

# Assumes scheduler isn't used
def load_checkpoint(checkpoint_path, device):
    checkpoint = torch.load(
        checkpoint_path, 
        map_location=device,
        weights_only=False
        )
    model = Transformer(**checkpoint['model_params'])
    model.load_state_dict(checkpoint['model_state_dict'])

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE
    )
    optimizer.load_state_dict(
        checkpoint['optimizer_state_dict']
        )
    return (
        model,
        checkpoint["tokenizer"],
        optimizer,
        checkpoint["epoch"],
        checkpoint["global_step"],
        checkpoint["val_loss"],
    )
