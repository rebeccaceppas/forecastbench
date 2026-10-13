"""Shared helpers: config loading, output folders, and access to the leaderboard code."""

import importlib.util
import json
import logging
import os
import shutil
import subprocess
import sys
import tomllib
import types
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
SRC_DIR = REPO_ROOT / "src"
LEADERBOARD_DIR = SRC_DIR / "leaderboard"
DEFAULT_CONFIG = HERE / "config.toml"

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("subset_accuracy")

_LEADERBOARD_MODULE = None


def load_config(path: Path = DEFAULT_CONFIG) -> Dict[str, Any]:
    """Read the TOML config and resolve data paths relative to this folder.

    Args:
      path (Path): Config file.
    """
    with open(path, "rb") as f:
        config = tomllib.load(f)
    for key in ("processed_forecasts_dir", "question_sets_dir", "cache_dir"):
        value = Path(config["data"][key])
        config["data"][key] = str(value if value.is_absolute() else HERE / value)
    config["_path"] = str(Path(path).resolve())
    return config


def _set_up_environment() -> None:
    """Set the environment variables the ForecastBench helpers read at import time."""
    os.environ.setdefault("RUNNING_LOCALLY", "1")
    variables_files = sorted(REPO_ROOT.glob("variables*.mk"))
    variables_files = [f for f in variables_files if f.name != "variables.example.mk"]
    if not variables_files:
        return
    for line in variables_files[0].read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if value.startswith("$"):
            value = os.environ.get(value[1:], "")
        os.environ.setdefault(key.strip(), value.strip())


def leaderboard():
    """Return `src/leaderboard/main.py` as a module, without editing or running it.

    Slack and git helpers fetch secrets at import time. This analysis never sends messages or
    commits, so both are replaced by stand-ins and no GCP credentials are needed to import.
    """
    global _LEADERBOARD_MODULE
    if _LEADERBOARD_MODULE is not None:
        return _LEADERBOARD_MODULE

    _set_up_environment()
    for path in (str(SRC_DIR), str(LEADERBOARD_DIR)):
        if path not in sys.path:
            sys.path.insert(0, path)

    import helpers  # noqa: E402

    def _send_message(message: str, *args, **kwargs) -> None:
        logger.warning(f"[slack stand-in] {message}")

    stubs = {
        "helpers.slack": types.SimpleNamespace(send_message=_send_message),
        "helpers.git": types.SimpleNamespace(),
    }
    for name, stub in stubs.items():
        sys.modules[name] = stub
        setattr(helpers, name.split(".")[1], stub)

    spec = importlib.util.spec_from_file_location("leaderboard_main", LEADERBOARD_DIR / "main.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["leaderboard_main"] = module
    spec.loader.exec_module(module)
    _LEADERBOARD_MODULE = module
    return module


def git_info() -> Dict[str, str]:
    """Return the current commit and whether the working tree has uncommitted changes."""

    def run(*args: str) -> str:
        try:
            return subprocess.run(
                ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
            ).stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return ""

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(run("status", "--porcelain", "--untracked-files=no")),
    }


def make_output_dir(config: Dict[str, Any], name: str) -> Path:
    """Create `outputs/<timestamp>_<name>/` and record the config and git commit in it.

    Args:
      config (Dict[str, Any]): Loaded config.
      name (str): Short run name appended to the folder.
    """
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out = HERE / "outputs" / f"{stamp}_{name}"
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    shutil.copy(config["_path"], out / "config.toml")
    with open(out / "run_info.json", "w") as f:
        json.dump({"git": git_info(), "created": stamp}, f, indent=2)
    logger.info(f"Writing outputs to {out}")
    return out
