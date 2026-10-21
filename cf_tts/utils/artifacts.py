"""Runtime locations for large CF-TTS artifacts.

The source checkout lives under ARTS home storage.  Model caches, downloaded
data, W&B artifacts, and temporary runtime state must instead live in the
nandrew9 scratch artifact root.  Call :func:`configure_artifact_environment`
from executable entry points before loading Torch Hub or Hugging Face assets.
"""

from __future__ import annotations

import os
from pathlib import Path


DEFAULT_ARTIFACT_ROOT = Path("/weka/scratch/jhu/nandrew9/xli257/linearvc")


def artifact_root() -> Path:
    """Return the persistent artifact root, honoring an explicit override."""
    return Path(os.environ.get("LINEARVC_ARTIFACT_ROOT", DEFAULT_ARTIFACT_ROOT)).expanduser()


def configure_artifact_environment() -> Path:
    """Set safe cache defaults without replacing explicit user or Slurm paths."""
    root = artifact_root()
    cache_root = root / ".cache" / "tts"
    os.environ.setdefault("LINEARVC_ARTIFACT_ROOT", str(root))
    os.environ.setdefault("TORCH_HOME", str(cache_root / "torch"))
    # Do not set HF_HOME: it also controls the cached `hf auth login` token.
    # Keep credentials in HOME while placing downloaded Hub data in scratch.
    os.environ.setdefault("HF_HUB_CACHE", str(cache_root / "huggingface" / "hub"))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.environ["HF_HUB_CACHE"])
    os.environ.setdefault("WANDB_DIR", str(cache_root / "wandb" / "runs"))
    os.environ.setdefault("WANDB_CACHE_DIR", str(cache_root / "wandb" / "cache"))
    os.environ.setdefault("WANDB_CONFIG_DIR", str(cache_root / "wandb" / "config"))
    # Do not override XDG_CACHE_HOME.  huggingface_hub derives HF_HOME (and
    # therefore the `hf auth login` token location) from it when HF_HOME is
    # unset.  HF_HUB_CACHE above is sufficient to keep the large Hub cache in
    # scratch while leaving the small credential in the user's home cache.
    os.environ.setdefault("PYTHONPYCACHEPREFIX", str(cache_root / "pycache"))
    return root
