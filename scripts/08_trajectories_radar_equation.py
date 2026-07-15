"""Stage 8: aircraft trajectories with SNR from the radar equation (R^-4)
but WITHOUT clutter and noise -- no fluctuation, no measurement noise, no
false alarms, no clutter.

With detection deterministic, the radar equation alone draws a hard
detection horizon: a crossing is recorded iff its mean SNR clears the CFAR
floor, i.e. iff its range is within

    R_horizon = range_ref * (snr_ref_lin / tau_lin)^(1/4)   (~74.8 km here)

so the output is the clean radar view of every trajectory out to that
range, and nothing beyond it. Stage 9 adds the stochastic layer on top.

Usage:
    python scripts/08_trajectories_radar_equation.py
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
from utils.plots import plot_bscope_coverage, plot_coverage, plot_rti_coverage
from utils.scenario import Scenario

PLOT_DAY_INDEX = 0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage 8: radar-equation SNR without clutter and noise.")
    parser.add_argument("--scenario", type=str, default=None)
    parser.add_argument("--input-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output dir (default: active/radar/stage08).")
    parser.add_argument("--threshold-min-db", type=float, default=None,
                        help="Override the scenario CFAR floor for this run "
                             "(e.g. 0 to extend the detection horizon).")
    return parser.parse_args()


def _fail(message: str) -> None:
    raise ValueError(f"Stage 08 validation failed: {message}")


def main() -> None:
    args = parse_args()
    sc = Scenario.load(args.scenario or get_scenario_path())
    if args.threshold_min_db is not None:
        sc.threshold_min_db = args.threshold_min_db
    output_dir = args.output_dir or get_stage_dir(8)

    day_files, scan_grid = ensure_beam_crossings(
        args.input_dir or get_trajectories_dir(), get_beam_crossings_dir(), sc)

    cfg = MeasurementConfig(snr_mode="radar_equation", fluctuation=False,
                            measurement_noise=False, false_alarms=False, clutter=False)
    results = run_days(day_files, scan_grid, sc, cfg, output_dir)

    # Deterministic detection horizon: SNR(R) == threshold.
    horizon_m = sc.range_ref_m * (10 ** (sc.snr_ref_db / 10) / sc.threshold_lin()) ** 0.25

    print("\n" + "=" * 70)
    print("VALIDATION GATE")
    print("=" * 70)
    for r in results:
        truth, dets = r["_truth"], r["_dets"]
        det, mis = truth[truth["detected"]], truth[~truth["detected"]]
        if len(det) and det["true_range_m"].max() > horizon_m + 1e-6:
            _fail(f"{r['date']}: detection beyond the deterministic horizon")
        if len(mis) and mis["true_range_m"].min() < horizon_m - 1e-6:
            _fail(f"{r['date']}: miss inside the deterministic horizon")
        if (dets["source"] != "target").any():
            _fail(f"{r['date']}: non-target detections present in the clean stage")
        if not np.allclose(dets["range_m"], dets["true_range_m"]):
            _fail(f"{r['date']}: measured range differs from truth without noise")
        if not np.allclose(truth["snr_db"], truth["snr_mean_db"], atol=1e-9):
            _fail(f"{r['date']}: snr_db deviates from the radar equation without fluctuation")
    print(f"  detection == (range <= horizon) exactly; targets only; "
          f"measurements == truth: OK")
    print(f"  deterministic detection horizon: {horizon_m / 1000:.1f} km "
          f"(SNR falls to the {sc.threshold_min_db:g} dB floor; instrumented range "
          f"{sc.range_max_m / 1000:.0f} km)")

    date, _ = day_files[PLOT_DAY_INDEX]
    truth0 = results[PLOT_DAY_INDEX]["_truth"]
    scan_t0, _ = scan_grid[date]
    plot_coverage(
        truth0, sc.range_max_m / 1000, horizon_m / 1000,
        f"Stage 8 PPI — truth vs detected ({date}, full day)\n"
        "aircraft ARE out to 200 km; the radar only detects inside the horizon",
        os.path.join(get_plot_dir(), f"6_PPI_{date}.png"))
    plot_bscope_coverage(
        truth0, sc.range_max_m / 1000,
        f"Stage 8 B-scope — truth vs detected ({date}, full day)\n"
        f"detected inside the {horizon_m/1000:.0f} km horizon; grey aircraft are there but unseen",
        os.path.join(get_plot_dir(), f"6_bscope_{date}.png"))
    plot_rti_coverage(
        truth0, scan_t0, sc.scan_period_s, sc.range_max_m / 1000,
        f"Stage 8 RTI — truth vs detected ({date}, full day)\n"
        f"detected inside the {horizon_m/1000:.0f} km horizon; grey aircraft are there but unseen",
        os.path.join(get_plot_dir(), f"6_RTI_{date}.png"))
    print(f"plots written to: {get_plot_dir()} (PPI, B-scope, RTI)")

    print("\n08_trajectories_radar_equation completed successfully.")


if __name__ == "__main__":
    main()
