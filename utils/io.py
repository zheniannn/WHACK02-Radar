"""Filesystem path helpers shared by both radar stages.

Same conventions as WHACK01-Preprocessing: paths resolve from the
repository's own location, and the data root defaults to a `data/`
directory next to the repository (override with WHACK_DATA_ROOT).

    <data root>/active/
    ├── trajectories_10s/   # WHACK01 stage 4 output (this repo's input)
    └── radar/              # everything this repo writes
"""

import os

# Repository root: one level above utils/.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_data_root() -> str:
    """$WHACK_DATA_ROOT if set, else `data/` beside the repository."""
    return os.environ.get("WHACK_DATA_ROOT") or os.path.join(os.path.dirname(_REPO_ROOT), "data")


def get_trajectories_dir(dt_s: float = 10.0) -> str:
    """WHACK01 stage-4 trajectory CSVs (this repo's ground-truth input)."""
    tag = f"{dt_s:g}".replace(".", "p") + "s"
    return os.path.join(get_data_root(), "active", f"trajectories_{tag}")


def get_radar_dir() -> str:
    """Root for everything the radar pipeline writes."""
    return os.path.join(get_data_root(), "active", "radar")


def get_scenario_path() -> str:
    """Radar scenario JSON (stage 5 output, stage 6 input)."""
    return os.path.join(get_radar_dir(), "scenario.json")


def get_measurements_dir() -> str:
    """Per-day detection and truth CSVs (stage 6 output)."""
    return os.path.join(get_radar_dir(), "measurements")


def get_measurements_summary_path(output_dir: str = "") -> str:
    """Cross-day measurement-generation summary CSV."""
    return os.path.join(output_dir or get_measurements_dir(), "radar_measurements_summary.csv")
