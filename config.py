# Optimization settings
NUM_EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 1e-3

# Experiment tracking settings
EXPERIMENT_TITLE = "main"

TRAINING_EXAMPLES_PER_METRIC_LOG = 5e4
GLOBAL_STEPS_PER_METRIC_LOG = max(TRAINING_EXAMPLES_PER_METRIC_LOG // BATCH_SIZE, 1)

ESTIMATE_WINRATE = False # Winrate is a very expensive metric, arguably unecessary for now
ESTIMATE_ILLEGAL_PROBABILITY_MASS = False # So far this hasn't proven itself as useful, turning off for now
GAMES_PER_WINRATE_ESTIMATE = 10

# Transformer settings
NUM_LAYERS = 6
NUM_HEADS = 6
D_MODEL = 144 # Must be a multiple of NUM_HEADS
# 692 = max number of moves (formally half-moves) in the dataset I am using
MAX_SEQUENCE_LENGTH = 3 * 692 + 1 # Must be of form 3k + 1 where k is a natural number 

# Tokenizer settings
PAD_TOKEN_ID = 0
BOS_TOKEN_ID = 1
UNK_TOKEN_ID = 2
SKIP_TOKEN_ID = 3

PAD_TOKEN = "<|PAD|>"
UNK_TOKEN = "<|UNK|>"
BOS_TOKEN = "<|BOS|>"
SKIP_TOKEN = "<|SKIP|>"

# Engine settings
ENGINE_PATH = "stockfish"
SECONDS_PER_ENGINE_MOVE = 0.1
ENGINE_ELO = 1350

# Paths
UCI_TRAIN_DATA_PATH = "data/train_set.parquet"
UCI_VAL_DATA_PATH = "data/val_set.parquet"
UCI_HOLDOUT_DATA_PATH = "data/holdout_set.parquet"
ARBITRARY_GAME_PATH = "data/arbitrary_games.jsonl"
