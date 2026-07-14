"""Data-derived A-scope: a REAL aircraft instead of hand-placed target marks.

Stage 5's default A-scope injects two illustrative targets at chosen ranges.
This one replaces them with a single real flight from 2022-06-06 --
N118AT, a Piper PA-44-180 Seminole (icao24 a049fd) -- shown at two scans of
its own outbound track: strong near (~25 km) and marginal far (~68 km),
14 minutes apart. Everything is from the data or the scenario physics: the
ranges/azimuths are the aircraft's true beam crossings, the echo power is
the radar-equation mean SNR at those ranges, real clutter patches falling in
the beam are drawn, and the noise is the scenario's Exp(1) floor.

Usage:
    python scripts/05b_real_ascope.py
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.beam_crossings import ensure_beam_crossings
from utils.io import get_beam_crossings_dir, get_plot_dir, get_scenario_path, get_trajectories_dir
from utils.plots import C_CLUTTER, C_TARGET, GRID, INK, INK2, MUTED
from utils.scenario import Scenario

DATE = "2022-06-06"
TRAJECTORY_ID = "a049fd_1654554529_r0"
AIRCRAFT = "N118AT  Piper PA-44-180 Seminole"
NEAR_KM, FAR_KM = 25.0, 68.0
FLOORS_DB = (8.0, 5.0)          # generate the A-scope at each CFAR floor


def _panel(ax, sc, r_km_target, snr_db, az_deg, scan_idx, label, seed, floor_db):
    rng = np.random.default_rng(seed)
    n = int((sc.range_max_m - sc.range_min_m) / sc.range_resolution_m)
    r_km = (sc.range_min_m + sc.range_resolution_m * (np.arange(n) + 0.5)) / 1000
    amp = rng.exponential(1.0, n)                        # noise floor
    ax.plot(r_km, np.maximum(10 * np.log10(amp), -20), color=MUTED, lw=0.6, zorder=2)

    # Real clutter patches whose azimuth falls in this beam.
    for p in sc.clutter_patches:
        if abs(((p["azimuth_deg"] - az_deg + 180) % 360) - 180) <= sc.azimuth_beamwidth_deg / 2:
            cdb = 10 * np.log10(1 + 10 ** (sc.clutter_snr_db / 10))
            ax.plot([p["range_m"] / 1000], [cdb], "o", ms=7, color=C_CLUTTER, zorder=5)
            ax.annotate("clutter", (p["range_m"] / 1000, cdb + 1.5), color=C_CLUTTER,
                        fontsize=8, ha="center")

    # The real aircraft: mean echo power at its true range.
    tdb = 10 * np.log10(1 + 10 ** (snr_db / 10))
    ax.plot([r_km_target], [tdb], "o", ms=8, color=C_TARGET, zorder=6)
    ax.annotate(f"N118AT\n{r_km_target:.0f} km · {tdb:.1f} dB", (r_km_target, tdb + 2),
                color=C_TARGET, fontsize=9, ha="center")

    ax.axhline(floor_db, color=INK, lw=1.3, ls="--", zorder=4)
    ax.annotate(f"CFAR floor {floor_db:g} dB", (2, floor_db + 0.7), color=INK, fontsize=8)
    ax.axhline(13.0, color=INK2, lw=1.1, ls=":", zorder=4)
    ax.annotate("conventional ~13 dB", (2, 13.7), color=INK2, fontsize=8)
    ax.set_xlim(0, sc.range_max_m / 1000 * 1.02); ax.set_ylim(-20, 32)
    ax.set_xlabel("range (km)")
    ax.set_title(label, color=INK, fontsize=11)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def plot_full_flight(sc, a, out_path):
    """The whole flight in one image: the aircraft's echo power vs range across
    every scan of its track, with the radar-equation mean curve and the CFAR
    floors. Shows the continuous fade and where the echo drops below each
    threshold -- the two side-by-side A-scopes are just two vertical slices
    of this."""
    rng = np.random.default_rng(sc.seed)
    r_km = a["true_range_m"].to_numpy() / 1000
    snr_lin = 10 ** (a["snr_mean_db"].to_numpy() / 10)
    mean_db = 10 * np.log10(1 + snr_lin)                    # mean echo power over noise
    z = rng.exponential(1 + snr_lin)                        # per-scan Swerling realisation
    draw_db = 10 * np.log10(z)

    order = np.argsort(r_km)
    rr, mm = r_km[order], mean_db[order]
    detected = draw_db >= 8.0

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.scatter(r_km[detected], draw_db[detected], s=14, color=C_TARGET, alpha=0.7,
               lw=0, zorder=3, label="detected (Swerling draw, ≥ 8 dB)")
    ax.scatter(r_km[~detected], draw_db[~detected], s=14, facecolor="none",
               edgecolor=MUTED, lw=0.7, zorder=3, label="missed (< 8 dB)")
    ax.plot(rr, mm, color=INK, lw=1.8, zorder=4, label="mean echo (radar equation)")

    for db, ls, txt in ((8.0, "--", "CFAR floor 8 dB"), (5.0, "--", "CFAR floor 5 dB"),
                        (13.0, ":", "conventional ~13 dB")):
        ax.axhline(db, color=INK2, lw=1.1, ls=ls, zorder=2)
        ax.annotate(txt, (sc.range_max_m / 1000 * 0.99, db + 0.6), color=INK2,
                    fontsize=8, ha="right")

    # Range at which the mean echo crosses each floor.
    for floor in (8.0, 5.0):
        below = np.where(mm < floor)[0]
        if below.size:
            rc = rr[below[0]]
            ax.axvline(rc, color=GRID, lw=1, zorder=1)
            ax.annotate(f"{rc:.0f} km", (rc, -18), color=INK2, fontsize=8, ha="center")

    ax.set_xlim(0, sc.range_max_m / 1000 * 1.02); ax.set_ylim(-20, 50)
    ax.set_xlabel("range (km)"); ax.set_ylabel("received power over mean noise (dB)")
    leg = ax.legend(loc="upper right", frameon=False, fontsize=9)
    for t in leg.get_texts():
        t.set_color(INK2)
    dur = (a["scan_idx"].max() - a["scan_idx"].min()) * sc.scan_period_s / 60
    ax.set_title(f"Full flight A-scope -- {AIRCRAFT} ({DATE})\n"
                 f"one aircraft, {len(a)} scans over {dur:.0f} min: echo fades with range "
                 "and drops below the floor near 75 km", color=INK)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150); plt.close(fig)


def main() -> None:
    sc = Scenario.load(get_scenario_path())
    ensure_beam_crossings(get_trajectories_dir(), get_beam_crossings_dir(), sc)
    cx = pd.read_csv(os.path.join(get_beam_crossings_dir(), f"beam_crossings_{DATE}.csv"))
    a = cx[cx.trajectory_id == TRAJECTORY_ID].sort_values("scan_idx")
    if a.empty:
        raise SystemExit(f"{TRAJECTORY_ID} not in {DATE} beam crossings")

    plot_full_flight(sc, a, os.path.join(get_plot_dir(), "stage05_ascope_full_flight.png"))
    print(f"full-flight A-scope -> {os.path.join(get_plot_dir(), 'stage05_ascope_full_flight.png')}")

    picks = []
    for target_km in (NEAR_KM, FAR_KM):
        row = a.loc[(a.true_range_m / 1000 - target_km).abs().idxmin()]
        picks.append(row)
    dt_min = (picks[1].scan_idx - picks[0].scan_idx) * sc.scan_period_s / 60

    for floor_db in FLOORS_DB:
        fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
        for ax, row, tag in zip(axes, picks, ("strong (near)", "marginal (far)")):
            _panel(ax, sc, row.true_range_m / 1000, row.snr_mean_db, row.true_azimuth_deg,
                   int(row.scan_idx),
                   f"scan {int(row.scan_idx)} · az {row.true_azimuth_deg:.0f}° · {tag}",
                   seed=sc.seed + int(row.scan_idx), floor_db=floor_db)
        axes[0].set_ylabel("received power over mean noise (dB)")
        fig.suptitle(f"A-scope from a real flight at a {floor_db:g} dB CFAR floor "
                     f"-- {AIRCRAFT} ({DATE})\n"
                     f"same aircraft {dt_min:.0f} min apart on its outbound track: "
                     f"strong at {picks[0].true_range_m/1000:.0f} km, "
                     f"marginal at {picks[1].true_range_m/1000:.0f} km as its echo fades",
                     color=INK, y=1.0)
        fig.tight_layout()
        out = os.path.join(get_plot_dir(), f"stage05_ascope_{floor_db:g}db.png")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        fig.savefig(out, dpi=150); plt.close(fig)
        print(f"real A-scope ({floor_db:g} dB) -> {out}")

    print(f"aircraft {AIRCRAFT}: near scan {int(picks[0].scan_idx)} "
          f"{picks[0].true_range_m/1000:.1f} km {picks[0].snr_mean_db:.1f} dB; "
          f"far scan {int(picks[1].scan_idx)} {picks[1].true_range_m/1000:.1f} km "
          f"{picks[1].snr_mean_db:.1f} dB")


if __name__ == "__main__":
    main()
