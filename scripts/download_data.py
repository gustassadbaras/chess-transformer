from datasets import load_dataset
import json

N = 100
output_path = f"data/subset_{N}.jsonl"

dataset = load_dataset("angeluriot/chess_games", split="train", streaming=True)
subset = list(dataset.take(N))

with open(output_path, "w", encoding="utf-8") as f:
    for game in subset:
        moves_uci = {"moves_uci" : game.get('moves_uci')}
        if moves_uci:
            f.write(json.dumps(moves_uci) + "\n")

print(f"Successfully generated '{output_path}' with {N} game examples.")
