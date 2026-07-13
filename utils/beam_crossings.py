"""Stage 6 rules: deterministic radar-truth geometry (no randomness).

For each trajectory and each scan, compute when the rotating beam crosses
the target, the true (slant range, azimuth, elevation) at that instant, the
coverage gate, and the mean SNR from the scenario's radar equation. Nothing
stochastic happens here -- detection draws, measurement noise, false alarms,
and clutter belong to stage 7, so this stage runs once and stage 7 can be
re-run cheaply (new seeds, new noise settings) on top of it.

Pipeline per day (see process_day):
  scan epochs on a fixed scan_period_s grid
  -> per trajectory: beam-crossing times (a rotating beam hits a target at
     scan_start + azimuth/360 * T; solved by two fixed-point iterations)
  -> truth position interpolated to those times (never extrapolated)
  -> coverage gating (range, elevation fan)
  -> mean SNR from the radar equation -> one row per in-coverage crossing.
"""

import os
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .geometry import enu_from_geodetic, polar_from_enu
from .scenario import Scenario

INPUT_PREFIX = "states_"
INPUT_SUFFIX = "_conventionalGA_trajectories_10s.csv"
DATE_PATTERN = re.compile(r"(\d{4}-\d{2}-\d{2})")

CROSSING_COLUMNS = [
    "date", "scan_idx", "t", "trajectory_id", "icao24",
    "true_range_m", "true_azimuth_deg", "true_elevation_deg", "snr_mean_db",
]


def discover_input_files(input_dir: str) -> List[Tuple[str, str]]:
    """Sorted (date, path) pairs for every stage-4 trajectory CSV in input_dir."""
    results = []
    for name in sorted(os.listdir(input_dir)):
        if not (name.startswith(INPUT_PREFIX) and name.endswith(INPUT_SUFFIX)):
            continue
        match = DATE_PATTERN.search(name)
        if not match:
            continue
        results.append((match.group(1), os.path.join(input_dir, name)))
    return results


def _bbox_prefilter(df: pd.DataFrame, sc: Scenario) -> pd.DataFrame:
    """Cheap lat/lon box cut before exact geometry: keeps only rows that can
    possibly be within range_max_m of the site (15% slack)."""
    half_deg = np.degrees(sc.range_max_m * 1.15 / 6_371_000.0)
    lat_ok = df["lat_interp"].sub(sc.site_lat_deg).abs() <= half_deg
    lon_ok = df["lon_interp"].sub(sc.site_lon_deg).abs() <= half_deg / max(
        np.cos(np.radians(sc.site_lat_deg)), 0.2)
    return df[lat_ok & lon_ok]


def _beam_crossing_states(tg, lat, lon, alt, scan_times, sc: Scenario):
    """Times and truth polar states at which the rotating beam crosses this
    trajectory, one per candidate scan.

    The beam points at azimuth az at time scan_start + az/360 * T, so the
    crossing time depends on the target's azimuth, which depends on its
    position at that time. GA targets move <1 km per scan, so two fixed-point
    iterations converge far below the measurement noise.

    Returns (t_hit, range_m, azimuth_deg, elevation_deg, valid_mask).
    """
    def polar_at(t):
        t_c = np.clip(t, tg[0], tg[-1])
        e, n, u = enu_from_geodetic(
            np.interp(t_c, tg, lat), np.interp(t_c, tg, lon), np.interp(t_c, tg, alt),
            sc.site_lat_deg, sc.site_lon_deg, sc.site_alt_m)
        return polar_from_enu(e, n, u)

    _, az0, _ = polar_at(scan_times)
    _, az1, _ = polar_at(scan_times + az0 / 360.0 * sc.scan_period_s)
    t_hit = scan_times + az1 / 360.0 * sc.scan_period_s
    rng_m, az, el = polar_at(t_hit)

    # Only crossings the trajectory actually spans (no extrapolation).
    valid = (t_hit >= tg[0]) & (t_hit <= tg[-1])
    return t_hit, rng_m, az, el, valid


def process_day(date: str, input_path: str, output_dir: str, sc: Scenario) -> Dict:
    """Compute one day's beam-crossing truth table. Returns the summary dict
    (including scan_t0/n_scans, which stage 7 needs to lay out false alarms)."""
    df = pd.read_csv(input_path, usecols=[
        "trajectory_id", "icao24", "timestamp", "lat_interp", "lon_interp", "alt_interp"])
    n_traj_day = df["trajectory_id"].nunique()
    df = _bbox_prefilter(df, sc)

    # Scan epochs cover the whole day, anchored on a multiple of the period.
    t_lo = np.floor(df["timestamp"].min() / sc.scan_period_s) * sc.scan_period_s if len(df) else 0.0
    t_hi = df["timestamp"].max() if len(df) else 0.0
    scan_times = np.arange(t_lo, t_hi + sc.scan_period_s, sc.scan_period_s)

    frames: List[pd.DataFrame] = []
    for tid, g in df.groupby("trajectory_id", sort=False):
        g = g.sort_values("timestamp")
        tg = g["timestamp"].to_numpy(float)
        if len(tg) < 2:
            continue
        lat, lon, alt = (g[c].to_numpy(float) for c in ("lat_interp", "lon_interp", "alt_interp"))

        k0 = np.searchsorted(scan_times, tg[0] - sc.scan_period_s)
        k1 = np.searchsorted(scan_times, tg[-1], side="right")
        cand = scan_times[k0:k1]
        if not len(cand):
            continue

        t_hit, rng_m, az, el, valid = _beam_crossing_states(tg, lat, lon, alt, cand, sc)
        covered = valid & (rng_m >= sc.range_min_m) & (rng_m <= sc.range_max_m) \
                        & (el >= sc.elevation_min_deg) & (el <= sc.elevation_max_deg)
        if not covered.any():
            continue

        idx = np.where(covered)[0]
        frames.append(pd.DataFrame({
            "date": date, "scan_idx": k0 + idx, "t": t_hit[idx],
            "trajectory_id": tid, "icao24": g["icao24"].iloc[0],
            "true_range_m": rng_m[idx], "true_azimuth_deg": az[idx],
            "true_elevation_deg": el[idx],
            "snr_mean_db": 10 * np.log10(sc.snr_mean_lin(rng_m[idx])),
        }))

    crossings = (pd.concat(frames, ignore_index=True) if frames
                 else pd.DataFrame(columns=CROSSING_COLUMNS))
    crossings = crossings[CROSSING_COLUMNS].sort_values(
        ["scan_idx", "t"], kind="mergesort").reset_index(drop=True)

    output_path = os.path.join(output_dir, f"beam_crossings_{date}.csv")
    crossings.to_csv(output_path, index=False)

    return {
        "date": date,
        "n_scans": len(scan_times),
        "scan_t0": float(t_lo),
        "trajectories_in_day": int(n_traj_day),
        "trajectories_in_coverage": int(crossings["trajectory_id"].nunique()),
        "crossings": len(crossings),
        "output_file": os.path.abspath(output_path),
        # for the validation gate only, not written to the summary CSV
        "_crossings": crossings,
    }
