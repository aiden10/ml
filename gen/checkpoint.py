from pathlib import Path

import torch

CHECKPOINT_DIR = Path(__file__).with_name("checkpoints")
DEFAULT_PATH = CHECKPOINT_DIR / "lm.pt"


def save_checkpoint(model, optimizer, step, path=DEFAULT_PATH, **extra) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "embed_dim": model.embed_dim,
            "vocab_size": len(model.vocab),
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            **extra,
        },
        path,
    )
    return path


def load_checkpoint(path=DEFAULT_PATH) -> dict:
    return torch.load(Path(path), map_location="cpu")


def restore(model, optimizer, checkpoint: dict) -> int:
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint.get("step", 0)
