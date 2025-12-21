"""
TCM (LIC_TCM) bootstrap utilities.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Iterable, Tuple


REPO_URL = "https://github.com/jmliu206/LIC_TCM.git"
DEFAULT_WEIGHTS: Tuple[Tuple[str, str], ...] = (
    ("1TK-CPiD2QwtWJqZoT_OyCtnxdQ7UNP56", "mse_lambda_0.05.pth.tar"),
    ("1zpkW_MCkUWl8nRUlza0L7Fk7dXlXciZd", "mse_lambda_0.0025.pth.tar"),
)


def _run(cmd: list[str], quiet: bool = False) -> None:
    kwargs = {"check": True}
    if quiet:
        kwargs["stdout"] = subprocess.DEVNULL
        kwargs["stderr"] = subprocess.DEVNULL
    subprocess.run(cmd, **kwargs)


def default_third_party_dir() -> Path:
    """Return `<repo>/third_party` for this project."""
    return Path(__file__).resolve().parent / "third_party"


def default_tcm_dir() -> Path:
    """Return `<repo>/third_party/LIC_TCM` (repo clone target)."""
    return default_third_party_dir() / "LIC_TCM"


def ensure_gdown() -> None:
    """Ensure `gdown` is available in current Python env."""
    try:
        _run([sys.executable, "-m", "gdown", "--version"], quiet=True)
    except Exception:
        _run([sys.executable, "-m", "pip", "install", "--upgrade", "gdown"], quiet=True)


def ensure_git() -> None:
    """Ensure git exists."""
    _run(["git", "--version"], quiet=True)


def ensure_tcm_repo(tcm_dir: Path, repo_url: str = REPO_URL) -> Path:
    """Clone LIC_TCM repo into `tcm_dir` if not present."""
    tcm_dir = Path(tcm_dir)
    tcm_dir.parent.mkdir(parents=True, exist_ok=True)
    if tcm_dir.exists():
        return tcm_dir
    ensure_git()
    _run(["git", "clone", repo_url, str(tcm_dir)])
    return tcm_dir


def ensure_tcm_weights(
    tcm_dir: Path,
    weights: Iterable[Tuple[str, str]] = DEFAULT_WEIGHTS,
) -> None:
    """Download missing checkpoint files into `tcm_dir`."""
    tcm_dir = Path(tcm_dir)
    ensure_gdown()
    for file_id, fname in weights:
        out = tcm_dir / fname
        if out.exists():
            continue
        _run([sys.executable, "-m", "gdown", file_id, "-O", str(out)])


def ensure_tcm_assets(
    tcm_dir: Path | None = None,
    repo_url: str = REPO_URL,
    weights: Iterable[Tuple[str, str]] = DEFAULT_WEIGHTS,
) -> Path:
    """
    Ensure LIC_TCM repo and checkpoint weights exist locally.

    Returns:
        Path to the LIC_TCM repo root directory.
    """
    tcm_dir = default_tcm_dir() if tcm_dir is None else Path(tcm_dir)
    ensure_tcm_repo(tcm_dir, repo_url=repo_url)
    ensure_tcm_weights(tcm_dir, weights=weights)
    return tcm_dir