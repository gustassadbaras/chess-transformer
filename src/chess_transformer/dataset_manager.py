"""
Each dataset has a dedicated direcotry inside data/datasets.
data/datasets/<name>/data.parquet - the primary data file. Other scripts generally assume this file is a kind of 'interface' to this dataset, its core
... (additional info to be added)
"""
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

class DatasetDirectory:
    ROOT = Path("data/datasets")

    def __init__(self, name: str, **metadata):
        self.name = name
        self.path = self.ROOT / name

        if not self.exists():
            self.path.mkdir(parents=True, exist_ok=True)

            initial_metadata = {
                "name": name,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "description": "",
                **self._git_info(),
                **metadata,
            }

            self._write_metadata(initial_metadata)

    @staticmethod
    def _git_info():
        def git(*args):
            try:
                return subprocess.check_output(
                    ["git", *args],
                    stderr=subprocess.DEVNULL,
                    text=True,
                ).strip() or None
            except (subprocess.CalledProcessError, FileNotFoundError):
                return None

        return {
            "commit": git("rev-parse", "HEAD"),
            "branch": git("branch", "--show-current"),
            "git_dirty": bool(git("status", "--porcelain")),
        }

    def exists(self) -> bool:
        return self.path.is_dir()

    def data_path(self) -> Path:
        return self.path / "data.parquet"

    def cache_path(self, name_with_extension: str) -> Path:
        path = self.path / "cache"
        path.mkdir(parents=True, exist_ok=True)
        return path / f"{name_with_extension}"

    def partitions_path(self) -> Path:
        path = self.path / "partitions"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def metadata_path(self) -> Path:
        return self.path / "metadata.json"

    def metadata(self) -> dict:
        with self.metadata_path().open("r", encoding="utf-8") as f:
            return json.load(f)

    def update_metadata(self, **kwargs):
        metadata = self.metadata()
        metadata.update(kwargs)
        self._write_metadata(metadata)

    def _write_metadata(self, metadata: dict):
        path = self.metadata_path()
        tmp_path = path.with_suffix(".tmp")

        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4, default=str)
            f.write("\n")

        tmp_path.replace(path)
