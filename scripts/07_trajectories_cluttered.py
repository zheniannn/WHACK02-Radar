"""Stage 7: aircraft trajectories at fixed SNR (15 dB) WITH clutter and
noise -- Swerling-1 fluctuation, measurement noise, Poisson false alarms,
and persistent clutter are all on, but the SNR is range-independent, so
detection quality is uniform across the coverage (Pd ~ 0.82 everywhere at
the 8 dB floor). The difference from stage 6's plot is purely the added
contamination; the difference from stage 8 is purely the radar equation.

Usage:
    python scripts/07_trajectories_cluttered.py
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

# Make utils/ importable regardless of the caller's working directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.beam_crossings import ensure_beam_crossings
from utils.io import get_beam_crossings_dir, get_plot_dir, get_scenario_path, get_stage_dir, get_trajectories_dir
from utils.measurements import MeasurementConfig, run_days
from utils.plots import densest_window, plot_bscope, plot_detection_window, plot_rti
from utils.scenario import Scenario

PLOT_DAY_INDEX = 0


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 7: fixed-SNR trajectories with clutter and noise.")
    parser.add_argument("--scenario", type=str, default=None)
    parser.add_argument("--input-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output dir (default: active/radar/stage07).")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override the scenario seed (Monte-Carlo repetitions).")
    return parser.parse_args()


def _fail(message: str) -> None:
    raise ValueError(f"Stage 07 validation failed: {message}")


def main() -> None:
    args = parse_args()
    sc = Scenario.load(args.scenario or get_scenario_path())
    if args.seed is not None:
        sc.seed = args.seed
    output_dir = args.output_dir or get_stage_dir(7)

    day_files, scan_grid = ensure_beam_crossings(
        args.input_dir or get_trajectories_dir(), get_beam_crossings_dir(), sc)

    cfg = MeasurementConfig(snr_mode="fixed", fluctuation=True, measurement_noise=True,
                            false_alarms=True, clutter=True)
    results = run_days(day_files, scan_grid, sc, cfg, output_dir)

    print("\n" + "=" * 70)
    print("VALIDATION GATE")
    print("=" * 70)

    # Pd must be range-INDEPENDENT and equal to the fixed-SNR closed form.
    pd_expected = float(np.exp(-sc.threshold_lin() / (1 + 10 ** (sc.snr_ref_db / 10))))
    truth = pd.concat([r["_truth"] for r in results], ignore_index=True)
    print(f"  Pd vs range at fixed {sc.snr_ref_db:g} dB (expected {pd_expected:.3f} everywhere):")
    edges = np.linspace(sc.range_min_m, sc.range_max_m, 4)
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = truth[truth["true_range_m"].between(lo, hi)]
        if len(sel) < 500:
            continue
        pd_emp = sel["detected"].mean()
        status = "OK" if abs(pd_emp - pd_expected) < 0.02 else "FAIL"
        print(f"    {lo / 1000:5.0f}-{hi / 1000:5.0f} km: {pd_emp:.3f}  {status}")
        if status == "FAIL":
            _fail("Pd is not uniform at fixed SNR")

    expected_fa = sc.expected_false_alarms_per_scan()
    for r in results:
        tol = 5.0 * np.sqrt(expected_fa / r["n_scans"])
        if abs(r["fa_per_scan"] - expected_fa) > tol:
            _fail(f"{r['date']}: FA/scan {r['fa_per_scan']:.1f} vs expected {expected_fa:.1f}")
    print(f"  false-alarm rate ~= {expected_fa:.1f}/scan on every day: OK")

    dets = pd.concat([r["_dets"] for r in results], ignore_index=True)
    tg = dets[dets["source"] == "target"]
    r_std = (tg["range_m"] - tg["true_range_m"]).std()
    az_std = ((tg["azimuth_deg"] - tg["true_azimuth_deg"] + 180) % 360 - 180).std()
    if abs(r_std - sc.sigma_range_m) > 0.05 * sc.sigma_range_m or \
       abs(az_std - sc.sigma_azimuth_deg) > 0.05 * sc.sigma_azimuth_deg:
        _fail("measurement noise does not match scenario sigmas")
    print(f"  measurement noise sigmas reproduced ({r_std:.1f} m, {az_std:.3f} deg): OK")

    date, _ = day_files[PLOT_DAY_INDEX]
    # read the plot day's FULL detections from disk (in-memory _dets keeps
    # only targets to bound memory at long range); one day is cheap.
    dets0 = pd.read_csv(os.path.join(output_dir, f"radar_detections_{date}.csv"))
    scan_t0, _ = scan_grid[date]
    k0 = densest_window(os.path.join(get_beam_crossings_dir(), f"beam_crossings_{date}.csv"))
    plot_detection_window(
        dets0, k0, 90, sc.range_max_m / 1000,
        f"Stage 7 — fixed SNR {sc.snr_ref_db:g} dB with clutter and noise ({date}, 15 min)\n"
        f"same window as stage 6; Pd {pd_expected:.2f} uniformly, plus false alarms and clutter",
        os.path.join(get_plot_dir(), f"stage07_trajectories_{date}.png"))
    plot_bscope(
        dets0, k0, 90, sc.range_max_m / 1000,
        f"Stage 7 B-scope — fixed SNR {sc.snr_ref_db:g} dB with clutter and noise ({date}, 15 min)\n"
        "the radar's native frame: targets drift, clutter pins to a fixed cell",
        os.path.join(get_plot_dir(), f"stage07_bscope_{date}.png"))
    plot_rti(
        dets0, k0, 360, scan_t0, sc.scan_period_s, sc.range_max_m / 1000,
        f"Stage 7 RTI — fixed SNR {sc.snr_ref_db:g} dB with clutter and noise ({date}, 60 min)\n"
        "targets slope with range rate; clutter draws flat lines; noise speckles",
        os.path.join(get_plot_dir(), f"stage07_rti_{date}.png"))
    print(f"plots written to: {get_plot_dir()} (PPI, B-scope, RTI)")

    print("\n07_trajectories_cluttered completed successfully.")


if __name__ == "__main__":
    main()
