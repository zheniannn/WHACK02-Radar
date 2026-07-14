"""Stage 9: full physics -- SNR from the radar equation (R^-4), with
clutter and noise -- and the maximum-range analysis: how far out can the
radar still hold an aircraft trajectory before dropping it?

Two range limits are reported:
  * detection limit: the range where single-scan Pd falls to 0.5;
  * tracking limit: the range where half of the tracks contain a gap of
    >= GAP_SCANS consecutive missed scans (a simple track-drop proxy --
    most trackers coast only a couple of scans before deleting a track).

Usage:
    python scripts/09_radar_equation_cluttered.py
    python scripts/09_radar_equation_cluttered.py --seed 7 --output-dir mc_run_7/
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

# Make utils/ importable regardless of the caller's working directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.beam_crossings import ensure_beam_crossings
from utils.io import get_beam_crossings_dir, get_plot_dir, get_scenario_path, get_stage_dir, get_trajectories_dir
from utils.measurements import MeasurementConfig, run_days
from utils.plots import (
    densest_window,
    per_track_drop_table,
    plot_bscope,
    plot_detection_window,
    plot_max_range,
    plot_rti,
)
from utils.scenario import Scenario

PLOT_DAY_INDEX = 0
GAP_SCANS = 3          # consecutive misses treated as a broken track
MIN_CROSSINGS = 30     # only tracks with >= this many crossings enter the drop analysis


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 9: radar-equation SNR with clutter and noise, max-range analysis.")
    parser.add_argument("--scenario", type=str, default=None)
    parser.add_argument("--input-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output dir (default: active/radar/stage09).")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override the scenario seed (Monte-Carlo repetitions).")
    parser.add_argument("--threshold-min-db", type=float, default=None,
                        help="Override the scenario CFAR floor (WARNING: below ~6 dB the "
                             "false-alarm count explodes; use scripts/09b_zero_db_window.py "
                             "for the 0 dB illustration).")
    return parser.parse_args()


def _fail(message: str) -> None:
    raise ValueError(f"Stage 09 validation failed: {message}")


def _crossing_50(x_km: np.ndarray, y: np.ndarray, direction: str = "down") -> float:
    """First range (km) where y crosses 0.5 (linearly interpolated).
    direction="down" for curves falling through 0.5 (Pd vs range),
    "up" for curves rising through it (broken-track fraction vs range)."""
    for i in range(1, len(y)):
        crossed = (y[i - 1] >= 0.5 > y[i]) if direction == "down" else (y[i - 1] < 0.5 <= y[i])
        if crossed:
            f = (0.5 - y[i - 1]) / (y[i] - y[i - 1])
            return float(x_km[i - 1] + f * (x_km[i] - x_km[i - 1]))
    return float("nan")


def main() -> None:
    args = parse_args()
    sc = Scenario.load(args.scenario or get_scenario_path())
    if args.seed is not None:
        sc.seed = args.seed
    if args.threshold_min_db is not None:
        sc.threshold_min_db = args.threshold_min_db
    output_dir = args.output_dir or get_stage_dir(9)

    day_files, scan_grid = ensure_beam_crossings(
        args.input_dir or get_trajectories_dir(), get_beam_crossings_dir(), sc)

    cfg = MeasurementConfig(snr_mode="radar_equation", fluctuation=True, measurement_noise=True,
                            false_alarms=True, clutter=True)
    results = run_days(day_files, scan_grid, sc, cfg, output_dir)

    print("\n" + "=" * 70)
    print("VALIDATION GATE")
    print("=" * 70)
    truth = pd.concat([r["_truth"] for r in results], ignore_index=True)
    print("  Pd vs range (empirical | theory):")
    edges = np.linspace(sc.range_min_m, sc.range_max_m, 4)
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = truth[truth["true_range_m"].between(lo, hi)]
        if len(sel) < 500:
            continue
        pd_emp = sel["detected"].mean()
        pd_theory = float(sc.pd(sel["true_range_m"].to_numpy()).mean())  # mean over the bin, not Pd(median): robust to wide bins
        status = "OK" if abs(pd_emp - pd_theory) < 0.05 else "FAIL"
        print(f"    {lo / 1000:5.0f}-{hi / 1000:5.0f} km: {pd_emp:.3f} | {pd_theory:.3f}  {status}")
        if status == "FAIL":
            _fail("Pd deviates from the Swerling-1 closed form")

    expected_fa = sc.expected_false_alarms_per_scan()
    for r in results:
        tol = 5.0 * np.sqrt(expected_fa / r["n_scans"])
        if abs(r["fa_per_scan"] - expected_fa) > tol:
            _fail(f"{r['date']}: FA/scan {r['fa_per_scan']:.1f} vs expected {expected_fa:.1f}")
    print(f"  false-alarm rate ~= {expected_fa:.1f}/scan on every day: OK")

    # --- Maximum-range analysis ---
    print("\n" + "=" * 70)
    print("MAXIMUM-RANGE ANALYSIS")
    print("=" * 70)
    edges = np.linspace(sc.range_min_m, sc.range_max_m, 17)
    mid_km = (edges[:-1] + edges[1:]) / 2000
    pd_emp = np.array([truth.loc[truth["true_range_m"].between(lo, hi), "detected"].mean()
                       for lo, hi in zip(edges[:-1], edges[1:])])
    r50_emp = _crossing_50(mid_km, pd_emp)

    # Closed-form detection limit: Pd(R) = 0.5.
    tau = sc.threshold_lin()
    snr50_lin = tau / np.log(2.0) - 1.0
    r50_theory = sc.range_ref_m / 1000 * (10 ** (sc.snr_ref_db / 10) / snr50_lin) ** 0.25

    track_table = per_track_drop_table(truth, MIN_CROSSINGS, GAP_SCANS)
    mids, fracs = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = track_table[(track_table["r_median_m"] >= lo) & (track_table["r_median_m"] < hi)]
        if len(sel) >= 5:
            mids.append((lo + hi) / 2000)
            fracs.append(sel["dropped"].mean())
    drop50 = _crossing_50(np.array(mids), np.array(fracs), direction="up")

    print(f"  detection limit  (single-scan Pd = 0.5):        {r50_emp:.1f} km empirical "
          f"| {r50_theory:.1f} km closed-form")
    print(f"  tracking limit   (50% of tracks have a >={GAP_SCANS}-scan gap): {drop50:.1f} km")
    print(f"  instrumented range: {sc.range_max_m / 1000:.0f} km -- the radar equation, not the "
          f"instrumented range, is the effective limit")

    report = {
        "detection_limit_km_empirical": round(r50_emp, 1),
        "detection_limit_km_theory": round(r50_theory, 1),
        "tracking_limit_km": round(drop50, 1),
        "gap_scans": GAP_SCANS,
        "min_crossings": MIN_CROSSINGS,
        "instrumented_range_km": sc.range_max_m / 1000,
        "threshold_db": sc.threshold_min_db,
    }
    report_path = os.path.join(output_dir, "max_range_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  report written to: {os.path.abspath(report_path)}")

    # --- Plots ---
    date, _ = day_files[PLOT_DAY_INDEX]
    # read the plot day's FULL detections from disk (in-memory _dets keeps
    # only targets to bound memory at long range); one day is cheap.
    dets0 = pd.read_csv(os.path.join(output_dir, f"radar_detections_{date}.csv"))
    scan_t0, _ = scan_grid[date]
    k0 = densest_window(os.path.join(get_beam_crossings_dir(), f"beam_crossings_{date}.csv"))
    plot_detection_window(
        dets0, k0, 90, sc.range_max_m / 1000,
        f"Stage 9 — radar-equation SNR with clutter and noise ({date}, 15 min)\n"
        "same window as stages 6-8; distant tracks fade and contamination is on",
        os.path.join(get_plot_dir(), f"stage09_trajectories_{date}.png"))
    plot_bscope(
        dets0, k0, 90, sc.range_max_m / 1000,
        f"Stage 9 B-scope — radar-equation SNR with clutter and noise ({date}, 15 min)\n"
        "the radar's native frame: targets drift, clutter pins to a fixed cell",
        os.path.join(get_plot_dir(), f"stage09_bscope_{date}.png"))
    plot_rti(
        dets0, k0, 360, scan_t0, sc.scan_period_s, sc.range_max_m / 1000,
        f"Stage 9 RTI — radar-equation SNR with clutter and noise ({date}, 60 min)\n"
        "targets slope and fade with range; clutter draws flat lines; noise speckles",
        os.path.join(get_plot_dir(), f"stage09_rti_{date}.png"))
    plot_max_range(truth, track_table, sc, r50_emp, drop50, GAP_SCANS,
                   os.path.join(get_plot_dir(), "stage09_max_range.png"))
    print(f"plots written to: {get_plot_dir()} (PPI, B-scope, RTI, max-range)")

    print("\n09_radar_equation_cluttered completed successfully.")


if __name__ == "__main__":
    main()
