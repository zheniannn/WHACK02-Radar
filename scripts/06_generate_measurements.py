"""Stage 6: generate simulated 2D radar measurements (range, azimuth) from
the stage-4 ground-truth trajectories, per the frozen scenario.json.

Outputs per day: radar_truth_<date>.csv (every in-coverage beam crossing of
every trajectory, detected or not) and radar_detections_<date>.csv (what a
tracker sees: targets + false alarms + clutter, with truth linkage).
Measurements are recorded down to the scenario's CFAR floor, so any higher
threshold can be applied post-hoc by filtering on snr_db.

Usage:
    python scripts/06_generate_measurements.py
    python scripts/06_generate_measurements.py --scenario custom.json
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

# Make utils/ importable regardless of the caller's working directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.io import (
    get_measurements_dir,
    get_measurements_summary_path,
    get_scenario_path,
    get_trajectories_dir,
)
from utils.radar_sim import discover_input_files, process_day
from utils.scenario import Scenario

SUMMARY_COLUMNS = [
    "date", "n_scans", "trajectories_in_day", "trajectories_in_coverage",
    "opportunities", "mean_pd_at_floor",
    "det_target", "det_noise", "det_clutter", "fa_per_scan",
    "truth_file", "detections_file",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Generate simulated 2D radar measurements.")
    parser.add_argument("--scenario", type=str, default=None,
                        help="Scenario JSON (default: active/radar/scenario.json from stage 5).")
    parser.add_argument("--input-dir", type=str, default=None,
                        help="Directory of stage-4 trajectory CSVs (default: active/trajectories_10s).")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory for truth/detection CSVs (default: active/radar/measurements).")
    return parser.parse_args()


# =============================================================================
# Validation gate -- run after processing, before declaring success
# =============================================================================

def _fail(message: str) -> None:
    raise ValueError(f"Stage 06 validation failed: {message}")


def _check_false_alarm_rate(day_results, sc: Scenario) -> None:
    """Empirical FA/scan must match n_cells * Pfa(floor) within 5 sigma."""
    expected = sc.expected_false_alarms_per_scan()
    for r in day_results:
        n_scans = r["n_scans"]
        tol = 5.0 * np.sqrt(expected / n_scans)   # Poisson std of the per-scan mean
        if abs(r["fa_per_scan"] - expected) > tol:
            _fail(f"{r['date']}: FA/scan {r['fa_per_scan']:.1f} vs expected {expected:.1f} (tol {tol:.1f})")
    print(f"  false-alarm rate ~= {expected:.1f}/scan on every day: OK")


def _check_pd_vs_theory(day_results, sc: Scenario) -> None:
    """Empirical Pd in three range bins must track the Swerling-1 closed form."""
    truth = pd.concat([r["_truth"] for r in day_results], ignore_index=True)
    print("  Pd vs range (empirical | theory):")
    edges = np.linspace(sc.range_min_m, sc.range_max_m, 4)
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = truth[(truth["true_range_m"] >= lo) & (truth["true_range_m"] < hi)]
        if len(sel) < 500:
            continue
        pd_emp = sel["detected"].mean()
        pd_theory = float(sc.pd(sel["true_range_m"].median()))
        status = "OK" if abs(pd_emp - pd_theory) < 0.05 else "FAIL"
        print(f"    {lo / 1000:5.0f}-{hi / 1000:5.0f} km: {pd_emp:.3f} | {pd_theory:.3f}  {status}")
        if status == "FAIL":
            _fail(f"Pd deviates from theory in {lo / 1000:.0f}-{hi / 1000:.0f} km bin")


def _check_measurement_noise(day_results, sc: Scenario) -> None:
    """Target residuals (measured - true) must reproduce sigma_range / sigma_azimuth."""
    dets = pd.concat([r["_dets"] for r in day_results], ignore_index=True)
    tg = dets[dets["source"] == "target"]
    r_std = (tg["range_m"] - tg["true_range_m"]).std()
    d_az = (tg["azimuth_deg"] - tg["true_azimuth_deg"] + 180.0) % 360.0 - 180.0
    az_std = d_az.std()
    ok_r = abs(r_std - sc.sigma_range_m) < 0.05 * sc.sigma_range_m
    ok_a = abs(az_std - sc.sigma_azimuth_deg) < 0.05 * sc.sigma_azimuth_deg
    print(f"  range residual std {r_std:.1f} m (spec {sc.sigma_range_m}), "
          f"azimuth {az_std:.3f} deg (spec {sc.sigma_azimuth_deg}): "
          f"{'OK' if ok_r and ok_a else 'FAIL'}")
    if not (ok_r and ok_a):
        _fail("measurement noise does not match scenario sigmas")


def _check_coverage_bounds(day_results, sc: Scenario) -> None:
    """True positions must respect the coverage gates exactly."""
    truth = pd.concat([r["_truth"] for r in day_results], ignore_index=True)
    in_range = truth["true_range_m"].between(sc.range_min_m, sc.range_max_m).all()
    in_el = truth["true_elevation_deg"].between(sc.elevation_min_deg, sc.elevation_max_deg).all()
    if not (in_range and in_el):
        _fail("truth rows outside the coverage volume")
    print("  all opportunities within coverage gates: OK")


def main() -> None:
    args = parse_args()
    sc = Scenario.load(args.scenario or get_scenario_path())
    input_dir = args.input_dir or get_trajectories_dir()
    output_dir = args.output_dir or get_measurements_dir()
    os.makedirs(output_dir, exist_ok=True)

    day_files = discover_input_files(input_dir)
    if not day_files:
        raise FileNotFoundError(f"No stage-4 trajectory CSVs found in {input_dir}")

    day_results = []
    for i, (date, path) in enumerate(day_files):
        # Independent, reproducible stream per day.
        rng = np.random.default_rng(sc.seed + i)
        result = process_day(date, path, output_dir, sc, rng)
        day_results.append(result)
        print(f"\n--- {result['date']} ---")
        print(f"scans:                       {result['n_scans']}")
        print(f"trajectories in coverage:    {result['trajectories_in_coverage']} / {result['trajectories_in_day']}")
        print(f"opportunities (beam hits):   {result['opportunities']}")
        print(f"mean Pd at {sc.threshold_min_db:.0f} dB floor:      {result['mean_pd_at_floor']:.3f}")
        print(f"detections target/noise/clutter: {result['det_target']} / {result['det_noise']} / {result['det_clutter']}")
        print(f"detections file:             {result['detections_file']}")

    summary_rows = [{k: v for k, v in r.items() if k in SUMMARY_COLUMNS} for r in day_results]
    summary_df = pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS)
    summary_path = get_measurements_summary_path(output_dir)
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary written to: {os.path.abspath(summary_path)}")

    print("\n" + "=" * 70)
    print("VALIDATION GATE")
    print("=" * 70)
    _check_coverage_bounds(day_results, sc)
    _check_false_alarm_rate(day_results, sc)
    _check_pd_vs_theory(day_results, sc)
    _check_measurement_noise(day_results, sc)

    print("\n06_generate_measurements completed successfully.")


if __name__ == "__main__":
    main()
