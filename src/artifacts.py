import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from src.models.cnn import baseline_CNN, MCDropout_CNN

MODEL_REGISTRY: dict[str, type] = {
    "baseline_CNN":  baseline_CNN,
    "MCDropout_CNN": MCDropout_CNN,
}

ARTIFACTS_ROOT = Path("artifacts")


def _git_commit_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def make_run_dir(model_name: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = ARTIFACTS_ROOT / f"{model_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_run(
    run_dir: Path,
    model: nn.Module,
    history: list[dict],
    config: dict[str, Any],
) -> None:
    config.setdefault("git_commit", _git_commit_hash())

    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2, default=str)

    with open(run_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    torch.save(model.state_dict(), run_dir / "model.pt")


def load_run(run_dir: str | Path) -> tuple[nn.Module, list[dict], dict]:
    """Load a saved run. Returns (model on CPU, history, config)."""
    run_dir = Path(run_dir)

    with open(run_dir / "config.json") as f:
        config = json.load(f)

    with open(run_dir / "history.json") as f:
        history = json.load(f)

    model_class_name = config["model_class"]
    if model_class_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model class '{model_class_name}'. "
            f"Available: {list(MODEL_REGISTRY)}"
        )

    model = MODEL_REGISTRY[model_class_name](**config.get("model_kwargs", {}))
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location="cpu", weights_only=True))
    model.eval()

    return model, history, config