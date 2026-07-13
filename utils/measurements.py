"""Stage 7 rules: the stochastic measurement layer on top of stage 6's
deterministic beam-crossing truth.

Per day (see process_day):
  read beam crossings (one row per in-coverage scan of each trajectory)
  -> Swerling-1 power draw per crossing; detected when it clears the CFAR
     floor (threshold_min_db)
  -> range/azimuth measurement noise on detected crossings
  -> Poisson false alarms over resolution cells x scans
  -> persistent clutter patches with per-scan fluctuation
  -> outputs: radar_truth_<date>.csv (every crossing + its measured SNR and
     detection outcome) and radar_detections_<date>.csv (what a tracker sees).

Detection statistics (square-law detector, exponential noise, Swerling 1):
  Pfa(tau)      = exp(-tau_lin)
  Pd(tau, snr)  = exp(-tau_lin / (1 + snr_lin)) = Pfa^(1/(1+snr_lin))
A cell's measured power z is Exp(1) for noise and Exp(1 + snr_lin) for a
target; "snr_db" in the outputs is 10*log10(z). Measurements are recorded
down to threshold_min_db, so any CFAR threshold >= that floor can be applied
post-hoc by filtering on snr_db -- one dataset supports a full ROC sweep.

Because geometry lives in stage 6, this stage can be re-run with different
seeds or noise settings (Monte Carlo) without recomputing beam crossings.
"""

import os
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .scenario import Scenario

INPUT_PREFIX = "beam_crossings_"
INPUT_SUFFIX = ".csv"
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
    """Sorted (date, path) pairs for every stage-6 beam-crossings CSV in input_dir."""
    results = []
    for name in sorted(os.listdir(input_dir)):
        if not (name.startswith(INPUT_PREFIX) and name.endswith(INPUT_SUFFIX)):
            continue
        match = DATE_PATTERN.search(name)
        if not match:
            continue
        results.append((match.group(1), os.path.join(input_dir, name)))
    return results


def load_scan_grid(summary_path: str) -> Dict[str, Tuple[float, int]]:
    """Read stage 6's summary to recover each day's scan grid (t0, n_scans) --
    needed to lay false alarms and clutter over scans with no targets."""
    if not os.path.exists(summary_path):
        raise FileNotFoundError(
            f"Stage-6 summary not found: {summary_path} (run 06_beam_crossings.py first)")
    s = pd.read_csv(summary_path)
    return {row["date"]: (float(row["scan_t0"]), int(row["n_scans"])) for _, row in s.iterrows()}


def _wrap_az(az_deg: np.ndarray) -> np.ndarray:
    return np.mod(az_deg, 360.0)


def process_day(date: str, crossings_path: str, output_dir: str, sc: Scenario,
                scan_t0: float, n_scans: int, rng: np.random.Generator) -> Dict:
    """Generate one day's truth and detection tables. Returns the summary dict."""
    cx = pd.read_csv(crossings_path, dtype={"trajectory_id": str, "icao24": str})
    tau_lin = sc.threshold_lin()
    scan_times = scan_t0 + sc.scan_period_s * np.arange(n_scans)

    det_frames: List[pd.DataFrame] = []

    # --- Targets: Swerling-1 draw per crossing, noise on detected ones ---
    snr_mean_lin = sc.snr_mean_lin(cx["true_range_m"].to_numpy())
    z = rng.exponential(1.0 + snr_mean_lin)
    detected = z >= tau_lin

    truth = cx.copy()
    truth["snr_db"] = 10 * np.log10(z)
    truth["detected"] = detected

    d = truth[detected]
    det_frames.append(pd.DataFrame({
        "date": date, "scan_idx": d["scan_idx"], "t": d["t"],
        "range_m": d["true_range_m"] + rng.normal(0.0, sc.sigma_range_m, len(d)),
        "azimuth_deg": _wrap_az(d["true_azimuth_deg"] + rng.normal(0.0, sc.sigma_azimuth_deg, len(d))),
        "snr_db": d["snr_db"], "source": "target",
        "trajectory_id": d["trajectory_id"], "icao24": d["icao24"],
        "true_range_m": d["true_range_m"], "true_azimuth_deg": d["true_azimuth_deg"],
    }))

    # --- False alarms: Poisson over cells x scans; conditional power is
    # memoryless (z = tau + Exp(1) given z > tau for exponential noise). ---
    n_fa = rng.poisson(sc.expected_false_alarms_per_scan() * n_scans)
    fa_scan = rng.integers(0, n_scans, n_fa)
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
        zc = rng.exponential(1.0 + clutter_snr_lin, n_scans)
        hit = np.where(zc >= tau_lin)[0]
        if not hit.size:
            continue
        det_frames.append(pd.DataFrame({
            "date": date, "scan_idx": hit,
            "t": scan_times[hit] + patch["azimuth_deg"] / 360.0 * sc.scan_period_s,
            "range_m": patch["range_m"] + rng.normal(0.0, sc.sigma_range_m, hit.size),
            "azimuth_deg": _wrap_az(patch["azimuth_deg"] + rng.normal(0.0, sc.sigma_azimuth_deg, hit.size)),
            "snr_db": 10 * np.log10(zc[hit]), "source": "clutter",
            "trajectory_id": "", "icao24": "",
            "true_range_m": patch["range_m"], "true_azimuth_deg": patch["azimuth_deg"],
        }))

    dets = pd.concat(det_frames, ignore_index=True)
    truth = truth[TRUTH_COLUMNS].sort_values(["scan_idx", "t"], kind="mergesort").reset_index(drop=True)
    dets = dets[DETECTION_COLUMNS].sort_values(["scan_idx", "t"], kind="mergesort").reset_index(drop=True)

    truth_path = os.path.join(output_dir, f"radar_truth_{date}.csv")
    det_path = os.path.join(output_dir, f"radar_detections_{date}.csv")
    truth.to_csv(truth_path, index=False)
    dets.to_csv(det_path, index=False)

    by_source = dets["source"].value_counts()
    return {
        "date": date,
        "n_scans": n_scans,
        "trajectories_in_coverage": int(truth["trajectory_id"].nunique()),
        "opportunities": len(truth),
        "mean_pd_at_floor": float(truth["detected"].mean()) if len(truth) else float("nan"),
        "det_target": int(by_source.get("target", 0)),
        "det_noise": int(by_source.get("noise", 0)),
        "det_clutter": int(by_source.get("clutter", 0)),
        "fa_per_scan": float(by_source.get("noise", 0) / max(n_scans, 1)),
        "truth_file": os.path.abspath(truth_path),
        "detections_file": os.path.abspath(det_path),
        # for the validation gate only, not written to the summary CSV
        "_truth": truth,
        "_dets": dets,
    }
