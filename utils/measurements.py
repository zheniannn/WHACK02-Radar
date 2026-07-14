"""Measurement generation shared by stages 6-8: each stage is one
MeasurementConfig applied to the shared beam-crossing geometry.

  Stage 6: fixed SNR, no fluctuation, no measurement noise, no false
           alarms, no clutter -- the pure radar view of the trajectories.
  Stage 7: fixed SNR + Swerling fluctuation + measurement noise + false
           alarms + clutter -- stochasticity at range-independent SNR.
  Stage 8: radar-equation SNR + everything -- full physics; distant
           targets fade and the maximum usable range emerges.

Detection statistics (square-law detector, exponential noise, Swerling 1):
  Pfa(tau)      = exp(-tau_lin)
  Pd(tau, snr)  = exp(-tau_lin / (1 + snr_lin)) = Pfa^(1/(1+snr_lin))
A cell's measured power z is Exp(1) for noise and Exp(1 + snr_lin) for a
target; "snr_db" in the outputs is 10*log10(z). Measurements are recorded
down to threshold_min_db, so any CFAR threshold >= that floor can be
applied post-hoc by filtering on snr_db.

With fluctuation disabled (stage 6) z is the mean SNR itself, so detection
is deterministic and snr_db is exact.
"""

import os
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .scenario import Scenario

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


@dataclass
class MeasurementConfig:
    """Which phenomena a stage turns on."""
    snr_mode: str = "radar_equation"   # "fixed" (snr_ref_db everywhere) | "radar_equation"
    fluctuation: bool = True           # Swerling-1 power draw (off -> deterministic detection)
    measurement_noise: bool = True     # Gaussian range/azimuth jitter
    false_alarms: bool = True          # Poisson noise detections over CFAR cells
    clutter: bool = True               # persistent scenario clutter patches


def _wrap_az(az_deg: np.ndarray) -> np.ndarray:
    return np.mod(az_deg, 360.0)


def process_day(date: str, crossings_path: str, output_dir: str, sc: Scenario,
                cfg: MeasurementConfig, scan_t0: float, n_scans: int,
                rng: np.random.Generator) -> Dict:
    """Generate one day's truth and detection tables under cfg. Returns the
    summary dict."""
    cx = pd.read_csv(crossings_path, dtype={"trajectory_id": str, "icao24": str})
    tau_lin = sc.threshold_lin()
    scan_times = scan_t0 + sc.scan_period_s * np.arange(n_scans)

    det_frames: List[pd.DataFrame] = []

    # --- Targets ---
    if cfg.snr_mode == "fixed":
        snr_mean_lin = np.full(len(cx), 10.0 ** (sc.snr_ref_db / 10.0))
    else:
        snr_mean_lin = sc.snr_mean_lin(cx["true_range_m"].to_numpy())

    if cfg.fluctuation:
        z = rng.exponential(1.0 + snr_mean_lin)     # Swerling 1 + noise, per crossing
    else:
        z = snr_mean_lin                            # noiseless: measured power is the mean
    detected = z >= tau_lin

    truth = cx.copy()
    truth["snr_mean_db"] = 10 * np.log10(snr_mean_lin)
    truth["snr_db"] = 10 * np.log10(z)
    truth["detected"] = detected

    d = truth[detected]
    sig_r = sc.sigma_range_m if cfg.measurement_noise else 0.0
    sig_a = sc.sigma_azimuth_deg if cfg.measurement_noise else 0.0
    det_frames.append(pd.DataFrame({
        "date": date, "scan_idx": d["scan_idx"], "t": d["t"],
        "range_m": d["true_range_m"] + (rng.normal(0.0, sig_r, len(d)) if sig_r else 0.0),
        "azimuth_deg": _wrap_az(d["true_azimuth_deg"] + (rng.normal(0.0, sig_a, len(d)) if sig_a else 0.0)),
        "snr_db": d["snr_db"], "source": "target",
        "trajectory_id": d["trajectory_id"], "icao24": d["icao24"],
        "true_range_m": d["true_range_m"], "true_azimuth_deg": d["true_azimuth_deg"],
    }))

    # --- False alarms: Poisson over cells x scans; conditional power is
    # memoryless (z = tau + Exp(1) given z > tau for exponential noise). ---
    if cfg.false_alarms:
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

    # --- Persistent clutter: fixed patches, fluctuating each scan, measured
    # with the same noise model. ---
    if cfg.clutter:
        clutter_snr_lin = 10.0 ** (sc.clutter_snr_db / 10.0)
        for patch in sc.clutter_patches:
            zc = rng.exponential(1.0 + clutter_snr_lin, n_scans)
            hit = np.where(zc >= tau_lin)[0]
            if not hit.size:
                continue
            det_frames.append(pd.DataFrame({
                "date": date, "scan_idx": hit,
                "t": scan_times[hit] + patch["azimuth_deg"] / 360.0 * sc.scan_period_s,
                "range_m": patch["range_m"] + (rng.normal(0.0, sig_r, hit.size) if sig_r else 0.0),
                "azimuth_deg": _wrap_az(patch["azimuth_deg"] + (rng.normal(0.0, sig_a, hit.size) if sig_a else 0.0)),
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
        "scan_t0": scan_t0,
        "trajectories_in_coverage": int(truth["trajectory_id"].nunique()),
        "opportunities": len(truth),
        "mean_pd": float(truth["detected"].mean()) if len(truth) else float("nan"),
        "det_target": int(by_source.get("target", 0)),
        "det_noise": int(by_source.get("noise", 0)),
        "det_clutter": int(by_source.get("clutter", 0)),
        "fa_per_scan": float(by_source.get("noise", 0) / max(n_scans, 1)),
        "truth_file": os.path.abspath(truth_path),
        "detections_file": os.path.abspath(det_path),
        # for validation gates / plots only, not written to the summary CSV
        "_truth": truth,
        # only target detections retained for the validation gate; the full
        # detection set (incl. millions of false alarms at long range) is on disk.
        "_dets": dets[dets["source"] == "target"].copy(),
    }


SUMMARY_COLUMNS = [
    "date", "n_scans", "trajectories_in_coverage", "opportunities", "mean_pd",
    "det_target", "det_noise", "det_clutter", "fa_per_scan",
    "truth_file", "detections_file",
]


def run_days(day_files: List[Tuple[str, str]], scan_grid: Dict, sc: Scenario,
             cfg: MeasurementConfig, output_dir: str) -> List[Dict]:
    """Process every day under cfg (per-day seeded RNG) and write the summary."""
    os.makedirs(output_dir, exist_ok=True)
    results = []
    for i, (date, path) in enumerate(day_files):
        scan_t0, n_scans = scan_grid[date]
        rng = np.random.default_rng(sc.seed + i)   # independent, reproducible per day
        r = process_day(date, path, output_dir, sc, cfg, scan_t0, n_scans, rng)
        results.append(r)
        print(f"\n--- {r['date']} ---")
        print(f"opportunities (beam crossings): {r['opportunities']}")
        print(f"mean Pd:                        {r['mean_pd']:.3f}")
        print(f"detections target/noise/clutter: {r['det_target']} / {r['det_noise']} / {r['det_clutter']}")

    summary = pd.DataFrame([{k: v for k, v in r.items() if k in SUMMARY_COLUMNS} for r in results],
                           columns=SUMMARY_COLUMNS)
    summary_path = os.path.join(output_dir, "measurements_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"\nSummary written to: {os.path.abspath(summary_path)}")
    return results
