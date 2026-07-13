"""Stage 5: choose the radar site from ground-truth traffic density and
freeze the full radar scenario (coverage, accuracy, SNR model, CFAR floor,
clutter map) into scenario.json for stage 6.

Usage:
    python scripts/05_radar_scenario.py
    python scripts/05_radar_scenario.py --range-max-km 60 --threshold-min-db 10
"""

import argparse
import os
import sys

import numpy as np

# Make utils/ importable regardless of the caller's working directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.io import get_scenario_path, get_trajectories_dir
from utils.radar_sim import discover_input_files
from utils.scenario import Scenario, generate_clutter_patches, select_site


def parse_args():
    parser = argparse.ArgumentParser(description="Select radar site and write scenario.json.")
    parser.add_argument("--input-dir", type=str, default=None,
                        help="Directory of stage-4 trajectory CSVs (default: active/trajectories_10s).")
    parser.add_argument("--output", type=str, default=None,
                        help="Scenario JSON path (default: active/radar/scenario.json).")
    parser.add_argument("--range-max-km", type=float, default=80.0,
                        help="Instrumented range in km (default: 80).")
    parser.add_argument("--threshold-min-db", type=float, default=8.0,
                        help="CFAR floor in dB; measurements are recorded down to this (default: 8).")
    parser.add_argument("--seed", type=int, default=20220606,
                        help="RNG seed frozen into the scenario (default: 20220606).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir or get_trajectories_dir()
    output_path = args.output or get_scenario_path()

    day_files = discover_input_files(input_dir)
    if not day_files:
        raise FileNotFoundError(f"No stage-4 trajectory CSVs found in {input_dir}")
    print(f"Selecting site from {len(day_files)} day(s) of trajectories...")

    site = select_site([path for _, path in day_files])
    sc = Scenario(
        site_lat_deg=site["site_lat_deg"],
        site_lon_deg=site["site_lon_deg"],
        site_alt_m=site["site_alt_m"],
        range_max_m=args.range_max_km * 1000.0,
        threshold_min_db=args.threshold_min_db,
        seed=args.seed,
    )
    sc.clutter_patches = generate_clutter_patches(sc, np.random.default_rng(sc.seed))
    sc.save(output_path)

    print(f"site: lat {sc.site_lat_deg}, lon {sc.site_lon_deg}, alt {sc.site_alt_m} m "
          f"(densest cell: {site['density_samples']} samples ~ "
          f"{site['density_samples'] * 10 / 3600:.0f} flight-hours)")
    print(f"coverage: {sc.range_min_m / 1000:.0f}-{sc.range_max_m / 1000:.0f} km, "
          f"elevation {sc.elevation_min_deg}-{sc.elevation_max_deg} deg, scan {sc.scan_period_s}s")
    print(f"cells/scan: {sc.n_cells()}, CFAR floor: {sc.threshold_min_db} dB "
          f"(expected false alarms/scan: {sc.expected_false_alarms_per_scan():.1f})")
    print(f"clutter patches: {len(sc.clutter_patches)}")

    # Radar equation, evaluated at the scenario's calibration (see Scenario.snr_mean_lin).
    print(f"\nradar equation ({sc.rcs_ref_m2:g} m^2 target, "
          f"{sc.snr_ref_db:g} dB @ {sc.range_ref_m / 1000:g} km, R^-4):")
    print(f"  {'range':>8} | {'mean SNR':>9} | {'Pd @ floor':>10}")
    for r_km in (20, 40, 60, 80):
        snr_db = 10 * np.log10(sc.snr_mean_lin(r_km * 1000.0))
        print(f"  {r_km:>5} km | {snr_db:>6.1f} dB | {float(sc.pd(r_km * 1000.0)):>10.3f}")

    print(f"\nscenario written to: {os.path.abspath(output_path)}")


if __name__ == "__main__":
    main()
