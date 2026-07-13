"""Stage 6 rules: generate simulated 2D radar measurements from ground-truth
trajectories.

Per day (see process_day):
  scan epochs on a fixed scan_period_s grid
  -> per trajectory: beam-crossing times (a rotating beam hits a target at
     scan_start + azimuth/360 * T; solved by two fixed-point iterations)
  -> truth position interpolated to those times; coverage gating
     (range, elevation fan)
  -> Swerling-1 SNR draw per opportunity; measurement recorded when it
     clears the CFAR floor (threshold_min_db)
  -> range/azimuth measurement noise
  -> Poisson false alarms + persistent clutter patches, drawn from the
     same detection statistics
  -> two outputs: a truth/opportunity table (every in-coverage scan of every
     trajectory, detected or not) and a detection table (what a tracker
     would actually see, with truth linkage for evaluation).

Detection statistics (square-law detector, exponential noise, Swerling 1):
  Pfa(tau)      = exp(-tau_lin)
  Pd(tau, snr)  = exp(-tau_lin / (1 + snr_lin)) = Pfa^(1/(1+snr_lin))
A cell's measured power z is Exp(1) for noise and Exp(1 + snr_lin) for a
target; "snr_db" in the outputs is 10*log10(z). Measurements are recorded
down to threshold_min_db, so any CFAR threshold >= that floor can be applied
post-hoc by filtering on snr_db -- one dataset supports a full ROC sweep.
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

DETECTION_COLUMNS = [
    "date", "scan_idx", "t", "range_m", "azimuth_deg", "snr_db", "source",
    "trajectory_id", "icao24", "true_range_m", "true_azimuth_deg",
]
TRUTH_COLUMNS = [
    "date", "scan_idx", "t", "trajectory_id", "icao24",
    "true_range_m", "true_azimuth_deg", "true_elevation_deg",
    "snr_mean_db", "snr_db", "detected",
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


def _wrap_az(az_deg: np.ndarray) -> np.ndarray:
    return np.mod(az_deg, 360.0)


def process_day(date: str, input_path: str, output_dir: str, sc: Scenario,
                rng: np.random.Generator) -> Dict:
    """Generate one day's truth and detection tables. Returns the summary dict."""
    df = pd.read_csv(input_path, usecols=[
        "trajectory_id", "icao24", "timestamp", "lat_interp", "lon_interp", "alt_interp"])
    n_traj_day = df["trajectory_id"].nunique()
    df = _bbox_prefilter(df, sc)

    # Scan epochs cover the whole day, anchored on a multiple of the period.
    t_lo = np.floor(df["timestamp"].min() / sc.scan_period_s) * sc.scan_period_s if len(df) else 0.0
    t_hi = df["timestamp"].max() if len(df) else 0.0
    scan_times = np.arange(t_lo, t_hi + sc.scan_period_s, sc.scan_period_s)

    tau_lin = sc.threshold_lin()

    truth_frames: List[pd.DataFrame] = []
    det_frames: List[pd.DataFrame] = []

    # --- Targets ---
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
        r, a, e_deg, th = rng_m[idx], az[idx], el[idx], t_hit[idx]
        snr_mean_lin = sc.snr_mean_lin(r)          # radar equation (see Scenario)
        z = rng.exponential(1.0 + snr_mean_lin)    # Swerling 1 + noise, per scan
        detected = z >= tau_lin

        truth_frames.append(pd.DataFrame({
            "date": date, "scan_idx": k0 + idx, "t": th,
            "trajectory_id": tid, "icao24": g["icao24"].iloc[0],
            "true_range_m": r, "true_azimuth_deg": a, "true_elevation_deg": e_deg,
            "snr_mean_db": 10 * np.log10(snr_mean_lin), "snr_db": 10 * np.log10(z),
            "detected": detected,
        }))
        if detected.any():
            d = np.where(detected)[0]
            det_frames.append(pd.DataFrame({
                "date": date, "scan_idx": k0 + idx[d], "t": th[d],
                "range_m": r[d] + rng.normal(0.0, sc.sigma_range_m, d.size),
                "azimuth_deg": _wrap_az(a[d] + rng.normal(0.0, sc.sigma_azimuth_deg, d.size)),
                "snr_db": 10 * np.log10(z[d]), "source": "target",
                "trajectory_id": tid, "icao24": g["icao24"].iloc[0],
                "true_range_m": r[d], "true_azimuth_deg": a[d],
            }))

    # --- False alarms: Poisson over cells x scans; conditional power is
    # memoryless (z = tau + Exp(1) given z > tau for exponential noise). ---
    lam_per_scan = sc.expected_false_alarms_per_scan()
    n_fa = rng.poisson(lam_per_scan * len(scan_times))
    fa_scan = rng.integers(0, len(scan_times), n_fa)
    fa_az = rng.uniform(0.0, 360.0, n_fa)
    det_frames.append(pd.DataFrame({
        "date": date, "scan_idx": fa_scan,
        "t": scan_times[fa_scan] + fa_az / 360.0 * sc.scan_period_s,
        "range_m": rng.uniform(sc.range_min_m, sc.range_max_m, n_fa),
        "azimuth_deg": fa_az,
        "snr_db": 10 * np.log10(tau_lin + rng.exponential(1.0, n_fa)),
        "source": "noise", "trajectory_id": "", "icao24": "",
        "true_range_m": np.nan, "true_azimuth_deg": np.nan,
    }))

    # --- Persistent clutter: fixed patches, Swerling-like fluctuation each
    # scan, positions jittered like real measurements. ---
    clutter_snr_lin = 10.0 ** (sc.clutter_snr_db / 10.0)
    for patch in sc.clutter_patches:
        z = rng.exponential(1.0 + clutter_snr_lin, len(scan_times))
        hit = np.where(z >= tau_lin)[0]
        if not hit.size:
            continue
        det_frames.append(pd.DataFrame({
            "date": date, "scan_idx": hit,
            "t": scan_times[hit] + patch["azimuth_deg"] / 360.0 * sc.scan_period_s,
            "range_m": patch["range_m"] + rng.normal(0.0, sc.sigma_range_m, hit.size),
            "azimuth_deg": _wrap_az(patch["azimuth_deg"] + rng.normal(0.0, sc.sigma_azimuth_deg, hit.size)),
            "snr_db": 10 * np.log10(z[hit]), "source": "clutter",
            "trajectory_id": "", "icao24": "",
            "true_range_m": patch["range_m"], "true_azimuth_deg": patch["azimuth_deg"],
        }))

    truth = (pd.concat(truth_frames, ignore_index=True) if truth_frames
             else pd.DataFrame(columns=TRUTH_COLUMNS))
    dets = (pd.concat(det_frames, ignore_index=True) if det_frames
            else pd.DataFrame(columns=DETECTION_COLUMNS))
    truth = truth[TRUTH_COLUMNS].sort_values(["scan_idx", "t"], kind="mergesort").reset_index(drop=True)
    dets = dets[DETECTION_COLUMNS].sort_values(["scan_idx", "t"], kind="mergesort").reset_index(drop=True)

    truth_path = os.path.join(output_dir, f"radar_truth_{date}.csv")
    det_path = os.path.join(output_dir, f"radar_detections_{date}.csv")
    truth.to_csv(truth_path, index=False)
    dets.to_csv(det_path, index=False)

    by_source = dets["source"].value_counts()
    return {
        "date": date,
        "n_scans": len(scan_times),
        "trajectories_in_day": int(n_traj_day),
        "trajectories_in_coverage": int(truth["trajectory_id"].nunique()),
        "opportunities": len(truth),
        "mean_pd_at_floor": float(truth["detected"].mean()) if len(truth) else float("nan"),
        "det_target": int(by_source.get("target", 0)),
        "det_noise": int(by_source.get("noise", 0)),
        "det_clutter": int(by_source.get("clutter", 0)),
        "fa_per_scan": float(by_source.get("noise", 0) / max(len(scan_times), 1)),
        "truth_file": os.path.abspath(truth_path),
        "detections_file": os.path.abspath(det_path),
        # for the validation gate only, not written to the summary CSV
        "_truth": truth,
        "_dets": dets,
    }
