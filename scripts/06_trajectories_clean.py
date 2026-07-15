"""Stage 6: the radar's view of the aircraft trajectories ALONE -- fixed
SNR of snr_ref_db (15 dB) everywhere, no fluctuation, no measurement noise,
no false alarms, no clutter. Detection is deterministic (15 dB clears the
8 dB floor on every crossing), so the output is the clean per-scan sampling
of every in-coverage trajectory, and the PPI plot shows pure tracks.

Usage:
    python scripts/06_trajectories_clean.py
"""

import argparse
import os
import sys

import numpy as np

# Make utils/ importable regardless of the caller's working directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.beam_crossings import ensure_beam_crossings
from utils.io import get_beam_crossings_dir, get_plot_dir, get_scenario_path, get_stage_dir, get_trajectories_dir
from utils.measurements import MeasurementConfig, run_days
from utils.plots import plot_bscope, plot_detection_window, plot_rti
from utils.scenario import Scenario

PLOT_DAY_INDEX = 0   # figures use the first day; data files cover all days


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 6: clean trajectories at fixed SNR.")
    parser.add_argument("--scenario", type=str, default=None)
    parser.add_argument("--input-dir", type=str, default=None,
                        help="Stage-4 trajectory CSVs (default: active/trajectories_10s).")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output dir (default: active/radar/stage06).")
    return parser.parse_args()


def _fail(message: str) -> None:
    raise ValueError(f"Stage 06 validation failed: {message}")


def main() -> None:
    args = parse_args()
    sc = Scenario.load(args.scenario or get_scenario_path())
    output_dir = args.output_dir or get_stage_dir(6)

    day_files, scan_grid = ensure_beam_crossings(
        args.input_dir or get_trajectories_dir(), get_beam_crossings_dir(), sc)

    cfg = MeasurementConfig(snr_mode="fixed", fluctuation=False, measurement_noise=False,
                            false_alarms=False, clutter=False)
    results = run_days(day_files, scan_grid, sc, cfg, output_dir)

    print("\n" + "=" * 70)
    print("VALIDATION GATE")
    print("=" * 70)
    for r in results:
        truth, dets = r["_truth"], r["_dets"]
        if not truth["detected"].all():
            _fail(f"{r['date']}: a crossing went undetected (impossible at fixed 15 dB, no noise)")
        if (dets["source"] != "target").any():
            _fail(f"{r['date']}: non-target detections present in the clean stage")
        if not np.allclose(dets["range_m"], dets["true_range_m"]):
            _fail(f"{r['date']}: measured range differs from truth without noise")
        if not np.allclose(truth["snr_db"], sc.snr_ref_db, atol=1e-9):
            _fail(f"{r['date']}: snr_db is not the fixed {sc.snr_ref_db} dB")
    print(f"  every crossing detected, targets only, measurements == truth, "
          f"snr == {sc.snr_ref_db:g} dB: OK")

    date, _ = day_files[PLOT_DAY_INDEX]
    dets0 = results[PLOT_DAY_INDEX]["_dets"]
    scan_t0, _ = scan_grid[date]
    k0 = 0   # full day
    plot_detection_window(
        dets0, k0, None, sc.range_max_m / 1000,
        f"Stage 6 PPI — trajectories only, fixed SNR {sc.snr_ref_db:g} dB ({date}, full day)\n"
        "no clutter, no noise: the clean radar view",
        os.path.join(get_plot_dir(), f"4_PPI_{date}.png"))
    plot_bscope(
        dets0, k0, None, sc.range_max_m / 1000,
        f"Stage 6 B-scope — clean, fixed SNR {sc.snr_ref_db:g} dB ({date}, full day)\n"
        "target tracks only; the clean 'before' to stage 7's contaminated frame",
        os.path.join(get_plot_dir(), f"4_bscope_{date}.png"))
    plot_rti(
        dets0, k0, None, scan_t0, sc.scan_period_s, sc.range_max_m / 1000,
        f"Stage 6 RTI — clean, fixed SNR {sc.snr_ref_db:g} dB ({date}, full day)\n"
        "target tracks only; the clean 'before' to stage 7's contaminated frame",
        os.path.join(get_plot_dir(), f"4_RTI_{date}.png"))
    print(f"plots written to: {get_plot_dir()} (PPI, B-scope, RTI)")

    print("\n06_trajectories_clean completed successfully.")


if __name__ == "__main__":
    main()
