"""Stage 9 at a 0 dB CFAR floor -- the feasible subset of its figures.

A full 4-day stage 9 at 0 dB is infeasible: Pfa = e^-1 = 0.37 per cell gives
~117k false alarms/scan (~1 billion/day), which neither fits in memory nor
means anything -- a 0 dB single-scan detector fires on a third of all cells.
This script produces what CAN be made at 0 dB:

  * max_range  -- from the TRUTH detection pattern only (no false alarms
                  needed), so it is exact and full-day: the detection and
                  tracking limits shift outward with the 119 km horizon.
  * trajectories (PPI) and RTI -- one 15-min window, where the ~10.5M
                  false alarms in the window bury the targets: the saturated
                  scope that shows why single-scan detection fails at 0 dB.

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
from utils.plots import (
    densest_window, per_track_drop_table, plot_detection_window, plot_max_range, plot_rti,
)
from utils.scenario import Scenario

THRESHOLD_DB = 0.0
WINDOW_SCANS = 90
DATE = "2022-06-06"
SEED = 20220606
GAP_SCANS = 3
MIN_CROSSINGS = 30


def _crossing_50(x_km, y, direction="down"):
    for i in range(1, len(y)):
        crossed = (y[i - 1] >= 0.5 > y[i]) if direction == "down" else (y[i - 1] < 0.5 <= y[i])
        if crossed:
            f = (0.5 - y[i - 1]) / (y[i] - y[i - 1])
            return float(x_km[i - 1] + f * (x_km[i] - x_km[i - 1]))
    return float("nan")


def main() -> None:
    sc = Scenario.load(get_scenario_path())
    sc.threshold_min_db = THRESHOLD_DB
    tau = sc.threshold_lin()
    horizon_km = sc.range_ref_m * (10 ** (sc.snr_ref_db / 10) / tau) ** 0.25 / 1000
    fa_per_scan = sc.expected_false_alarms_per_scan()
    pdir = get_plot_dir()
    day_files, scan_grid = ensure_beam_crossings(get_trajectories_dir(), get_beam_crossings_dir(), sc)
    print(f"0 dB floor: horizon {horizon_km:.0f} km, {fa_per_scan:,.0f} false alarms/scan")

    # --- max_range: truth detections at 0 dB, all days (no false alarms needed) ---
    rng = np.random.default_rng(SEED)
    truth_frames = []
    for date, _ in day_files:
        cx = pd.read_csv(os.path.join(get_beam_crossings_dir(), f"beam_crossings_{date}.csv"))
        snr = sc.snr_mean_lin(cx["true_range_m"].to_numpy())
        z = rng.exponential(1.0 + snr)
        truth_frames.append(pd.DataFrame({
            "trajectory_id": cx["trajectory_id"], "scan_idx": cx["scan_idx"],
            "true_range_m": cx["true_range_m"], "detected": z >= tau}))
    truth = pd.concat(truth_frames, ignore_index=True)

    edges = np.linspace(sc.range_min_m, sc.range_max_m, 17)
    mid = (edges[:-1] + edges[1:]) / 2000
    pd_emp = np.array([truth.loc[truth["true_range_m"].between(lo, hi), "detected"].mean()
                       for lo, hi in zip(edges[:-1], edges[1:])])
    r50 = _crossing_50(mid, pd_emp)
    tt = per_track_drop_table(truth, MIN_CROSSINGS, GAP_SCANS)
    mids, fracs = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = tt[(tt["r_median_m"] >= lo) & (tt["r_median_m"] < hi)]
        if len(sel) >= 5:
            mids.append((lo + hi) / 2000); fracs.append(sel["dropped"].mean())
    drop50 = _crossing_50(np.array(mids), np.array(fracs), "up")
    print(f"  detection limit {r50:.1f} km, tracking limit {drop50:.1f} km (0 dB)")
    plot_max_range(truth, tt, sc, r50, drop50, GAP_SCANS,
                   os.path.join(pdir, "stage09_0db_max_range.png"))

    # --- one window with the false-alarm flood, for PPI + RTI ---
    cx_path = os.path.join(get_beam_crossings_dir(), f"beam_crossings_{DATE}.csv")
    k0 = densest_window(cx_path, WINDOW_SCANS); k1 = k0 + WINDOW_SCANS
    scan_t0, _ = scan_grid[DATE]
    scan_times = scan_t0 + sc.scan_period_s * np.arange(k0, k1)

    cx = pd.read_csv(cx_path)
    cx = cx[(cx["scan_idx"] >= k0) & (cx["scan_idx"] < k1)]
    zt = rng.exponential(1.0 + sc.snr_mean_lin(cx["true_range_m"].to_numpy()))
    det = cx[zt >= tau]
    frames = [pd.DataFrame({
        "scan_idx": det["scan_idx"].to_numpy(),
        "t": scan_t0 + det["scan_idx"].to_numpy() * sc.scan_period_s
             + det["true_azimuth_deg"].to_numpy() / 360 * sc.scan_period_s,
        "range_m": det["true_range_m"].to_numpy(),
        "azimuth_deg": det["true_azimuth_deg"].to_numpy(), "source": "target"})]

    n_fa = rng.poisson(fa_per_scan * WINDOW_SCANS)
    fa_scan = rng.integers(k0, k1, n_fa); fa_az = rng.uniform(0, 360, n_fa)
    frames.append(pd.DataFrame({
        "scan_idx": fa_scan,
        "t": scan_t0 + fa_scan * sc.scan_period_s + fa_az / 360 * sc.scan_period_s,
        "range_m": rng.uniform(sc.range_min_m, sc.range_max_m, n_fa),
        "azimuth_deg": fa_az, "source": "noise"}))

    clutter_lin = 10 ** (sc.clutter_snr_db / 10)
    for p in sc.clutter_patches:
        zc = rng.exponential(1 + clutter_lin, WINDOW_SCANS)
        hit = np.where(zc >= tau)[0]
        if hit.size:
            frames.append(pd.DataFrame({
                "scan_idx": k0 + hit,
                "t": scan_times[hit] + p["azimuth_deg"] / 360 * sc.scan_period_s,
                "range_m": p["range_m"],
                "azimuth_deg": _wrap_az(p["azimuth_deg"] + rng.normal(0, sc.sigma_azimuth_deg, hit.size)),
                "source": "clutter"}))
    dets = pd.concat(frames, ignore_index=True)
    print(f"  window: {int((dets['source']=='target').sum()):,} targets, "
          f"{int((dets['source']=='noise').sum()):,} false alarms")

    plot_detection_window(
        dets, k0, WINDOW_SCANS, sc.range_max_m / 1000,
        f"Stage 9 at 0 dB CFAR floor -- one 15-min window ({DATE})\n"
        f"targets reach {horizon_km:.0f} km but drown in ~{fa_per_scan/1000:.0f}k false alarms/scan",
        os.path.join(pdir, "stage09_0db_trajectories.png"), horizon_km=horizon_km)
    plot_rti(
        dets, k0, WINDOW_SCANS, scan_t0, sc.scan_period_s, sc.range_max_m / 1000,
        f"Stage 9 RTI at 0 dB CFAR floor -- 15-min window ({DATE})\n"
        "targets slope through a near-solid false-alarm field",
        os.path.join(pdir, "stage09_0db_rti.png"))
    print(f"plots -> {pdir} (stage09_0db_max_range, _trajectories, _rti)")


if __name__ == "__main__":
    main()
