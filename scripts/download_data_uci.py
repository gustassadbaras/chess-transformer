from datasets import load_dataset
import json
import sys
import random
import math

# Usage:
# python -m scripts.download_data
# <number of games to download> <train set fraction> <val set fraction> <test set fraction>
# e.g.
# python -m download_data_uci 10000 0.6 0.3 0.1
# NOTE: use only once (or with caution) so as to not 'contaminate' validation/test sets.

if len(sys.argv) != 5:
    print(
        "Usage: download_data_uci.py "
        "<number_of_games> <train_fraction> <val_fraction> <test_fraction>"
    )
    sys.exit(1)

N = int(sys.argv[1])
train_fraction, val_fraction, test_fraction = (
    float(arg) for arg in sys.argv[2:5]
)

if not math.isclose(
    train_fraction + val_fraction + test_fraction,
    1.0):
    print("Error: dataset fractions must sum to 1.")
    sys.exit(1)


output_path = f"data/subset_{N}.jsonl"

dataset = load_dataset("angeluriot/chess_games", split="train", streaming=True)
subset = list(dataset.take(N))
random.seed(42)
random.shuffle(subset)

train_end = int(N * train_fraction)
val_end = train_end + int(N * val_fraction)

train_set = subset[:train_end]
val_set = subset[train_end:val_end]
test_set = subset[val_end:]


for name, split in [
    ("train", train_set),
    ("val", val_set),
    ("test", test_set),
]:
    output_path = f"data/{name}.jsonl"

    with open(output_path, "w", encoding="utf-8") as f:
        for game in split:
            moves_uci = game.get("moves_uci")

            if moves_uci:
                f.write(
                    json.dumps({"moves_uci": moves_uci})
                    + "\n"
                )

print(
    f"Successfully generated dataset with "
    f"{len(train_set)} training, "
    f"{len(val_set)} validation, "
    f"{len(test_set)} test games."
)