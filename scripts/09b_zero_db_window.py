"""Stage 9 at a 0 dB CFAR floor -- ONE 15-minute window only.

A full 4-day stage 9 at 0 dB is infeasible: Pfa = e^-1 = 0.37 per cell gives
~117k false alarms/scan (~1 billion/day), which neither fits in memory nor
means anything -- a 0 dB single-scan detector fires on a third of all cells.
This script generates just one densest 15-minute window so the saturated
scope is visible: targets now reach the ~119 km 0 dB horizon, but they are
buried under a near-uniform false-alarm field. This is the extreme end of
the problem statement's "lower the threshold -> more targets AND far more
false detections" -- and why single-scan detection alone is hopeless here.

Usage:
    python scripts/09b_zero_db_window.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.beam_crossings import ensure_beam_crossings
from utils.io import get_beam_crossings_dir, get_plot_dir, get_scenario_path, get_trajectories_dir
from utils.measurements import _wrap_az
from utils.plots import densest_window, plot_detection_window
from utils.scenario import Scenario

THRESHOLD_DB = 0.0
WINDOW_SCANS = 90
DATE = "2022-06-06"
SEED = 20220606


def main() -> None:
    sc = Scenario.load(get_scenario_path())
    sc.threshold_min_db = THRESHOLD_DB
    tau = sc.threshold_lin()
    rng = np.random.default_rng(SEED)

    day_files, scan_grid = ensure_beam_crossings(get_trajectories_dir(), get_beam_crossings_dir(), sc)
    cx_path = os.path.join(get_beam_crossings_dir(), f"beam_crossings_{DATE}.csv")
    k0 = densest_window(cx_path, WINDOW_SCANS)
    k1 = k0 + WINDOW_SCANS
    scan_t0, _ = scan_grid[DATE]

    horizon_km = sc.range_ref_m * (10 ** (sc.snr_ref_db / 10) / tau) ** 0.25 / 1000
    fa_per_scan = sc.expected_false_alarms_per_scan()
    print(f"0 dB floor: horizon {horizon_km:.0f} km, {fa_per_scan:,.0f} false alarms/scan "
          f"({fa_per_scan * WINDOW_SCANS:,.0f} in the {WINDOW_SCANS}-scan window)")

    # --- Targets in the window (radar-equation SNR + Swerling draw) ---
    cx = pd.read_csv(cx_path)
    cx = cx[(cx["scan_idx"] >= k0) & (cx["scan_idx"] < k1)]
    snr_lin = sc.snr_mean_lin(cx["true_range_m"].to_numpy())
    z = rng.exponential(1.0 + snr_lin)
    det = cx[z >= tau]
    frames = [pd.DataFrame({
        "scan_idx": det["scan_idx"], "range_m": det["true_range_m"],
        "azimuth_deg": det["true_azimuth_deg"], "source": "target"})]

    # --- False alarms over the window (this is the flood) ---
    n_fa = rng.poisson(fa_per_scan * WINDOW_SCANS)
    frames.append(pd.DataFrame({
        "scan_idx": rng.integers(k0, k1, n_fa),
        "range_m": rng.uniform(sc.range_min_m, sc.range_max_m, n_fa),
        "azimuth_deg": rng.uniform(0, 360, n_fa), "source": "noise"}))

    # --- Clutter in the window ---
    clutter_lin = 10 ** (sc.clutter_snr_db / 10)
    scan_times = scan_t0 + sc.scan_period_s * np.arange(k0, k1)
    for p in sc.clutter_patches:
        zc = rng.exponential(1 + clutter_lin, WINDOW_SCANS)
        hit = np.where(zc >= tau)[0]
        if hit.size:
            frames.append(pd.DataFrame({
                "scan_idx": k0 + hit, "range_m": p["range_m"],
                "azimuth_deg": _wrap_az(p["azimuth_deg"] + rng.normal(0, sc.sigma_azimuth_deg, hit.size)),
                "source": "clutter"}))

    dets = pd.concat(frames, ignore_index=True)
    n_t = int((dets["source"] == "target").sum()); n_n = int((dets["source"] == "noise").sum())
    print(f"window: {n_t:,} target detections, {n_n:,} false alarms, "
          f"{int((dets['source']=='clutter').sum()):,} clutter")

    out = os.path.join(get_plot_dir(), "stage09_0db_window.png")
    plot_detection_window(
        dets, k0, WINDOW_SCANS, sc.range_max_m / 1000,
        f"Stage 9 at a 0 dB CFAR floor -- one 15-min window ({DATE})\n"
        f"targets reach {horizon_km:.0f} km but drown in ~{fa_per_scan/1000:.0f}k false alarms/scan",
        out, horizon_km=horizon_km)
    print(f"plot -> {out}")


if __name__ == "__main__":
    main()
