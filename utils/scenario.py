"""Stage 5 rules: radar site selection and scenario definition.

The site is placed at the centre of the densest traffic cell in the
ground-truth trajectories (maximising usable tracks per scan), the site
elevation is estimated from the lowest flight altitudes seen nearby (a
proxy for local terrain, since no DEM is used), and the full radar
parameter set -- coverage, accuracy, SNR model, CFAR floor, clutter map --
is frozen into scenario.json so stage 6 runs are reproducible.
"""

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

DENSITY_BIN_DEG = 0.25          # traffic-density histogram cell size
SITE_ALT_BOX_DEG = 0.5          # half-width of the box used to estimate site elevation
SITE_ALT_PERCENTILE = 1.0       # low-altitude percentile used as the terrain proxy
SITE_ALT_MARGIN_M = 150.0       # subtracted from the proxy (aircraft fly above terrain)


@dataclass
class Scenario:
    """Everything stage 6 needs, frozen at stage-5 time."""
    # Site (chosen from the data)
    site_lat_deg: float = 0.0
    site_lon_deg: float = 0.0
    site_alt_m: float = 0.0
    # Coverage (2D fan-beam surveillance radar)
    scan_period_s: float = 10.0
    range_min_m: float = 1_000.0
    range_max_m: float = 200_000.0
    elevation_min_deg: float = 0.3
    elevation_max_deg: float = 30.0
    # Resolution cells (set the CFAR false-alarm opportunity count)
    range_resolution_m: float = 150.0
    azimuth_beamwidth_deg: float = 1.5
    # Measurement accuracy
    sigma_range_m: float = 50.0
    sigma_azimuth_deg: float = 0.2
    # SNR model: mean SNR for RCS_REF at RANGE_REF, R^-4 falloff (radar equation)
    snr_ref_db: float = 15.0
    range_ref_m: float = 50_000.0
    rcs_ref_m2: float = 1.0          # all light GA modelled at 1 m^2 (Swerling 1 handles fluctuation)
    # Detection: measurements are recorded down to this CFAR floor; any
    # threshold >= this can be applied post-hoc by filtering on snr_db.
    threshold_min_db: float = 8.0
    # Persistent ground-clutter patches (fixed across days)
    clutter_n_patches: int = 25
    clutter_range_max_m: float = 40_000.0
    clutter_snr_db: float = 12.0
    clutter_patches: List[Dict] = field(default_factory=list)  # [{range_m, azimuth_deg}]
    # Reproducibility
    seed: int = 20220606

    def n_cells(self) -> int:
        """CFAR resolution cells per scan."""
        n_range = int((self.range_max_m - self.range_min_m) / self.range_resolution_m)
        n_az = int(round(360.0 / self.azimuth_beamwidth_deg))
        return n_range * n_az

    # --- Radar physics (the scenario owns the model; stage 6 only applies it) ---

    def threshold_lin(self, threshold_db: float = None) -> float:
        """CFAR threshold in linear power units (defaults to the recording floor)."""
        return 10.0 ** ((self.threshold_min_db if threshold_db is None else threshold_db) / 10.0)

    def snr_mean_lin(self, range_m) -> np.ndarray:
        """Radar equation in calibrated form: mean SNR (linear) at range_m for a
        rcs_ref_m2 target -- snr_ref at range_ref with R^-4 two-way falloff."""
        return 10.0 ** (self.snr_ref_db / 10.0) * (self.range_ref_m / np.asarray(range_m, float)) ** 4

    def pfa(self, threshold_db: float = None) -> float:
        """Per-cell false-alarm probability, square-law detector on exponential
        noise: Pfa = exp(-tau)."""
        return float(np.exp(-self.threshold_lin(threshold_db)))

    def pd(self, range_m, threshold_db: float = None) -> np.ndarray:
        """Swerling-1 detection probability at range_m:
        Pd = exp(-tau / (1 + snr)) = Pfa^(1/(1+snr))."""
        return np.exp(-self.threshold_lin(threshold_db) / (1.0 + self.snr_mean_lin(range_m)))

    def expected_false_alarms_per_scan(self, threshold_db: float = None) -> float:
        """n_cells * Pfa at the given threshold."""
        return self.n_cells() * self.pfa(threshold_db)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "Scenario":
        with open(path) as f:
            return cls(**json.load(f))


def accumulate_density(traj_path: str, counts: Dict) -> pd.DataFrame:
    """Read one day's trajectories (needed columns only) and add its samples
    to the running density histogram. Returns the day's dataframe for reuse."""
    df = pd.read_csv(traj_path, usecols=["trajectory_id", "lat_interp", "lon_interp", "alt_interp"])
    lat_bin = np.floor(df["lat_interp"] / DENSITY_BIN_DEG).astype(int)
    lon_bin = np.floor(df["lon_interp"] / DENSITY_BIN_DEG).astype(int)
    grouped = pd.DataFrame({"lat_bin": lat_bin, "lon_bin": lon_bin}).value_counts()
    for key, n in grouped.items():
        counts[key] = counts.get(key, 0) + int(n)
    return df


def select_site(day_paths: List[str]) -> Dict:
    """Pick the densest DENSITY_BIN_DEG cell across all days as the radar site.

    Density is sample-count (each sample = 10 s of flight), i.e. dwell time,
    so busy training areas outweigh single fast transits. Site elevation is
    the SITE_ALT_PERCENTILE of altitudes within SITE_ALT_BOX_DEG of the site
    minus SITE_ALT_MARGIN_M (terrain proxy; clamped at sea level).

    Returns {site_lat_deg, site_lon_deg, site_alt_m, density_samples,
             per_day_samples_in_cell}.
    """
    counts: Dict = {}
    frames = []
    for path in day_paths:
        frames.append(accumulate_density(path, counts))

    (lat_bin, lon_bin), n_samples = max(counts.items(), key=lambda kv: kv[1])
    site_lat = (lat_bin + 0.5) * DENSITY_BIN_DEG
    site_lon = (lon_bin + 0.5) * DENSITY_BIN_DEG

    # Terrain proxy from the lowest nearby flight altitudes.
    near_alts = []
    for df in frames:
        near = (df["lat_interp"].sub(site_lat).abs() <= SITE_ALT_BOX_DEG) & \
               (df["lon_interp"].sub(site_lon).abs() <= SITE_ALT_BOX_DEG)
        near_alts.append(df.loc[near, "alt_interp"])
    alt_proxy = float(np.percentile(pd.concat(near_alts), SITE_ALT_PERCENTILE))
    site_alt = max(0.0, alt_proxy - SITE_ALT_MARGIN_M)

    return {
        "site_lat_deg": round(site_lat, 4),
        "site_lon_deg": round(site_lon, 4),
        "site_alt_m": round(site_alt, 1),
        "density_samples": int(n_samples),
    }


def generate_clutter_patches(sc: Scenario, rng: np.random.Generator) -> List[Dict]:
    """Fixed ground-clutter patch positions (same across days -- clutter is
    stationary). Uniform in azimuth; uniform in range up to clutter_range_max_m."""
    return [
        {"range_m": float(r), "azimuth_deg": float(a)}
        for r, a in zip(
            rng.uniform(sc.range_min_m, sc.clutter_range_max_m, sc.clutter_n_patches),
            rng.uniform(0.0, 360.0, sc.clutter_n_patches),
        )
    ]
