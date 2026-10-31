# Optimization settings
NUM_EPOCHS = 100
BATCH_SIZE = 1024
LEARNING_RATE = 1e-3

# Experiment tracking settings
EXPERIMENT_TITLE = "board-state"

TRAINING_EXAMPLES_PER_METRIC_LOG = 5e4
GLOBAL_STEPS_PER_METRIC_LOG = max(TRAINING_EXAMPLES_PER_METRIC_LOG // BATCH_SIZE, 1)

ESTIMATE_WINRATE = False
ESTIMATE_ILLEGAL_PROBABILITY_MASS = False
GAMES_PER_WINRATE_ESTIMATE = 10

# Transformer settings
NUM_LAYERS = 8
NUM_HEADS = 8
D_MODEL = 256 # Must be a multiple of NUM_HEADS
BOARD_SEQUENCE_LENGTH = 70

# Engine settings
ENGINE_PATH = "stockfish"
SECONDS_PER_ENGINE_MOVE = 0.1
ENGINE_ELO = 1350

# Paths
TRAIN_DATA_NAME = "combined_train"
VAL_DATA_NAME = "combined_val"
DATA_DIR = "data"

RANDOM_GAME_PATH = "data/processed/random_games.jsonl"

# Token IDs
CLS_ID = 25