from pathlib import Path

from hydra.core.hydra_config import HydraConfig


def output_path(name: str) -> Path:
    """Return a path inside the active Hydra invocation directory."""
    return Path(HydraConfig.get().runtime.output_dir) / name
