# Optimization settings
NUM_EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 1e-3

# Experiment tracking settings
EXPERIMENT_TITLE = "board-state"

TRAINING_EXAMPLES_PER_METRIC_LOG = 5e4
GLOBAL_STEPS_PER_METRIC_LOG = max(TRAINING_EXAMPLES_PER_METRIC_LOG // BATCH_SIZE, 1)

ESTIMATE_WINRATE = False
ESTIMATE_ILLEGAL_PROBABILITY_MASS = False
GAMES_PER_WINRATE_ESTIMATE = 10

# Transformer settings
NUM_LAYERS = 6
NUM_HEADS = 6
D_MODEL = 144 # Must be a multiple of NUM_HEADS
MAX_SEQUENCE_LENGTH = 66

# Engine settings
ENGINE_PATH = "stockfish"
SECONDS_PER_ENGINE_MOVE = 0.1
ENGINE_ELO = 1350

# Paths
UCI_TRAIN_DATA_PATH = "data/train_set.parquet"
UCI_VAL_DATA_PATH = "data/val_set.parquet"
UCI_HOLDOUT_DATA_PATH = "data/holdout_set.parquet"
RANDOM_GAME_PATH = "data/random_games.jsonl"
