from pathlib import Path
from config import DATA_DIR, TRAIN_DATA_NAME, VAL_DATA_NAME

DATASET_NAMES = {
    "train": TRAIN_DATA_NAME,
    "val": VAL_DATA_NAME
}

STAGE_EXTENSIONS = {
    "relabeled": ".parquet",
    "processed": ".parquet",
    "cached": ".npy"
}

# cache_type: flat/offset
def get_path(dataset: str, stage: str, cache_type=None, temporary=False):
    try:
        directory = f"{DATA_DIR}/{stage}/"
        name = DATASET_NAMES[dataset]
        extension = STAGE_EXTENSIONS[stage]
    except KeyError as e:
        raise ValueError(f"Unknown dataset or stage: {e.args[0]}") from e

    temp_str = "temp_" if temporary else ""

    if cache_type is not None and stage != "cached":
        raise ValueError("cache_type can only be specified for the cache stage")

    if cache_type:
        return f"{directory}{temp_str}{name}_{cache_type}{extension}"

    return f"{directory}{name}{extension}"