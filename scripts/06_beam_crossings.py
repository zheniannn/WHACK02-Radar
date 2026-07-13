"""Stage 6: deterministic radar-truth geometry -- when the rotating beam
crosses each trajectory, the true range/azimuth/elevation at that instant,
and the radar-equation mean SNR. No randomness: detection draws, noise,
false alarms, and clutter are stage 7, which can be re-run (new seeds, new
noise settings) without recomputing this stage.

Usage:
    python scripts/06_beam_crossings.py
    python scripts/06_beam_crossings.py --scenario custom.json
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

# Make utils/ importable regardless of the caller's working directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.beam_crossings import discover_input_files, process_day
from utils.io import (
    get_beam_crossings_dir,
    get_beam_crossings_summary_path,
    get_scenario_path,
    get_trajectories_dir,
)
from utils.scenario import Scenario

SUMMARY_COLUMNS = [
    "date", "n_scans", "scan_t0", "trajectories_in_day",
    "trajectories_in_coverage", "crossings", "output_file",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Compute deterministic radar beam-crossing truth.")
    parser.add_argument("--scenario", type=str, default=None,
                        help="Scenario JSON (default: active/radar/scenario.json from stage 5).")
    parser.add_argument("--input-dir", type=str, default=None,
                        help="Directory of stage-4 trajectory CSVs (default: active/trajectories_10s).")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory for beam-crossing CSVs (default: active/radar/beam_crossings).")
    return parser.parse_args()


# =============================================================================
# Validation gate -- run after processing, before declaring success
# =============================================================================

def _fail(message: str) -> None:
    raise ValueError(f"Stage 06 validation failed: {message}")


def _check_coverage_bounds(day_results, sc: Scenario) -> None:
    """Every crossing must respect the coverage gates exactly."""
    for r in day_results:
        cx = r["_crossings"]
        if not cx["true_range_m"].between(sc.range_min_m, sc.range_max_m).all():
            _fail(f"{r['date']}: crossing outside range gate")
        if not cx["true_elevation_deg"].between(sc.elevation_min_deg, sc.elevation_max_deg).all():
            _fail(f"{r['date']}: crossing outside elevation gate")
    print("  all crossings within coverage gates: OK")


def _check_snr_matches_radar_equation(day_results, sc: Scenario) -> None:
    """snr_mean_db must be exactly the scenario's radar equation at true_range_m."""
    for r in day_results:
        cx = r["_crossings"]
        expected = 10 * np.log10(sc.snr_mean_lin(cx["true_range_m"].to_numpy()))
        if not np.allclose(cx["snr_mean_db"].to_numpy(), expected, atol=1e-9):
            _fail(f"{r['date']}: snr_mean_db deviates from the radar equation")
    print("  snr_mean_db == radar equation at true range: OK")


def _check_beam_timing(day_results, sc: Scenario) -> None:
    """Every crossing must fall inside its own scan's rotation window:
    0 <= t - scan_start < T. (Gap SIZES between crossings are not bounded --
    a target passing near the radar legitimately swings the beam-timing
    offset by a large fraction of a scan.)"""
    T = sc.scan_period_s
    for r in day_results:
        cx = r["_crossings"]
        offset = cx["t"] - (r["scan_t0"] + cx["scan_idx"] * T)
        if not ((offset >= 0.0) & (offset < T)).all():
            _fail(f"{r['date']}: crossing outside its scan's rotation window")
    print("  every crossing inside its scan's rotation window: OK")


def main() -> None:
    args = parse_args()
    sc = Scenario.load(args.scenario or get_scenario_path())
    input_dir = args.input_dir or get_trajectories_dir()
    output_dir = args.output_dir or get_beam_crossings_dir()
    os.makedirs(output_dir, exist_ok=True)

    day_files = discover_input_files(input_dir)
    if not day_files:
        raise FileNotFoundError(f"No stage-4 trajectory CSVs found in {input_dir}")

    day_results = []
    for date, path in day_files:
        result = process_day(date, path, output_dir, sc)
        day_results.append(result)
        print(f"\n--- {result['date']} ---")
        print(f"scans:                    {result['n_scans']}")
        print(f"trajectories in coverage: {result['trajectories_in_coverage']} / {result['trajectories_in_day']}")
        print(f"beam crossings:           {result['crossings']}")
        print(f"output file:              {result['output_file']}")

    summary_rows = [{k: v for k, v in r.items() if k in SUMMARY_COLUMNS} for r in day_results]
    summary_df = pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS)
    summary_path = get_beam_crossings_summary_path(output_dir)
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary written to: {os.path.abspath(summary_path)}")

    print("\n" + "=" * 70)
    print("VALIDATION GATE")
    print("=" * 70)
    _check_coverage_bounds(day_results, sc)
    _check_snr_matches_radar_equation(day_results, sc)
    _check_beam_timing(day_results, sc)

    print("\n06_beam_crossings completed successfully.")


if __name__ == "__main__":
    main()
