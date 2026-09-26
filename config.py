# Experiment tracking settings
EXPERIMENT_TITLE = "single-token-per-move"

EPOCHS_PER_SAVE = 5 # Might want this to align with estimate frequencies

ESTIMATE_WINRATE = True
EPOCHS_PER_WINRATE_ESTIMATE = 3
GAMES_PER_WINRATE_ESTIMATE = 5

ESTIMATE_ILLEGAL_PROBABILITY_MASS = True
EPOCHS_PER_ILLEGAL_PROBABILITY_MASS_ESTIMATE = 3

EPOCHS_PER_GRAD_NORM_LOG = 5

# Optimization settings
NUM_EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 1e-3

# Transformer settings
NUM_LAYERS = 4
NUM_HEADS = 4 
D_MODEL = 128 # Must be a multiple of NUM_HEADS
MAX_SEQUENCE_LENGTH = 512

# Tokenizer settings
PAD_TOKEN_ID = 0
BOS_TOKEN_ID = 1
UNK_TOKEN_ID = 2

PAD_TOKEN = "<|PAD|>"
UNK_TOKEN = "<|UNK|>"
BOS_TOKEN = "<|BOS|>"

# Engine settings
ENGINE_PATH = "stockfish"
SECONDS_PER_ENGINE_MOVE = 0.1
ENGINE_ELO = 1350

# Paths
UCI_TRAIN_DATA_PATH = "data/train.jsonl"
UCI_VAL_DATA_PATH = "data/val.jsonl"
UCI_TEST_DATA_PATH = "data/test.jsonl"
RANDOM_GAME_PATH = "data/random_games.jsonl"
