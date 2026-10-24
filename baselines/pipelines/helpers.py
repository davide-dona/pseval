from pathlib import Path

from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

import pseval

# Lets a stage place its outputs beside the artifact it reads.
OmegaConf.register_new_resolver('parent', lambda path: str(Path(path).parent), replace=True)


def load_benchmark(dataset: str, root: str | None) -> pseval.Benchmark:
    """Load a benchmark from a local root, or from the Hub release of the installed pseval."""
    return pseval.load_benchmark(dataset, root=None if root is None else Path(root))


def output_path(name: str) -> Path:
    """Return a path inside the active Hydra invocation directory."""
    return Path(HydraConfig.get().runtime.output_dir) / name
