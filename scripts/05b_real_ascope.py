"""Data-derived detection figures for one real flight.

Uses a single real 2022-06-06 flight -- N118AT, a Piper PA-44-180 Seminole
(icao24 a049fd), outbound 8 -> 200 km -- and the scenario physics to show
where the radar can and cannot hold it:

  stage05_ascope_<floor>db_distance.png  echo power vs range across the whole
      flight (mean radar-equation curve + per-scan Swerling draws marked
      detected/missed), against the Exp(1) noise floor, at 8 dB and 5 dB.
  stage05_flight.png                     the aircraft's ground track on a PPI,
      blue inside the detection horizon and grey beyond.

Ranges/azimuths are the aircraft's true beam crossings; echo power is the
radar-equation mean SNR at those ranges; noise is the scenario's Exp(1) floor.

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
from utils.plots import C_NOISE, C_TARGET, GRID, INK, INK2, MUTED
from utils.scenario import Scenario

DATE = "2022-06-06"
TRAJECTORY_ID = "a049fd_1654554529_r0"
AIRCRAFT = "N118AT  Piper PA-44-180 Seminole"
FLOORS_DB = (8.0, 5.0)          # generate the A-scope at each CFAR floor


def plot_full_flight(sc, a, out_path, floor_db):
    """The whole flight in one image: the aircraft's echo power vs range
    (distance) across every scan of its track, with the radar-equation mean
    curve and the CFAR floor at floor_db. Per-scan Swerling draws are marked
    detected/missed against floor_db, so the 5 dB and 8 dB versions differ in
    how far out the aircraft stays detectable."""
    rng = np.random.default_rng(sc.seed)
    r_km = a["true_range_m"].to_numpy() / 1000
    snr_lin = 10 ** (a["snr_mean_db"].to_numpy() / 10)
    mean_db = 10 * np.log10(1 + snr_lin)                    # mean echo power over noise
    z = rng.exponential(1 + snr_lin)                        # per-scan Swerling realisation
    draw_db = 10 * np.log10(z)

    order = np.argsort(r_km)
    rr, mm = r_km[order], mean_db[order]
    detected = draw_db >= floor_db

    # Noise cells: Exp(1) power, range-independent, spread across the display.
    # These are the background the echo competes against; the ones above the
    # floor are the false alarms that the CFAR floor admits.
    n_noise = 5000
    noise_r = rng.uniform(sc.range_min_m / 1000, sc.range_max_m / 1000, n_noise)
    noise_db = 10 * np.log10(rng.exponential(1.0, n_noise))
    fa = noise_db >= floor_db

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.scatter(noise_r[~fa], noise_db[~fa], s=3, color=C_NOISE, alpha=0.18, lw=0,
               zorder=1, label="noise cells")
    ax.scatter(noise_r[fa], noise_db[fa], s=8, color=C_NOISE, alpha=0.7, lw=0,
               zorder=2, label=f"noise false alarms (≥ {floor_db:g} dB): {int(fa.sum())}/{n_noise}")
    ax.scatter(r_km[detected], draw_db[detected], s=16, color=C_TARGET, alpha=0.8,
               lw=0, zorder=4, label=f"aircraft detected (≥ {floor_db:g} dB)")
    ax.scatter(r_km[~detected], draw_db[~detected], s=16, facecolor="none",
               edgecolor=C_TARGET, lw=0.8, zorder=4, label=f"aircraft missed (< {floor_db:g} dB)")
    ax.plot(rr, mm, color=INK, lw=1.8, zorder=5, label="mean echo (radar equation)")

    ax.axhline(floor_db, color=INK, lw=1.4, ls="--", zorder=2)
    ax.annotate(f"CFAR floor {floor_db:g} dB", (sc.range_max_m / 1000 * 0.99, floor_db + 0.6),
                color=INK, fontsize=9, ha="right")
    ax.axhline(13.0, color=INK2, lw=1.1, ls=":", zorder=2)
    ax.annotate("conventional ~13 dB", (sc.range_max_m / 1000 * 0.99, 13.6),
                color=INK2, fontsize=8, ha="right")

    # Range at which the mean echo crosses this floor (the detection horizon).
    below = np.where(mm < floor_db)[0]
    if below.size:
        rc = rr[below[0]]
        ax.axvline(rc, color=GRID, lw=1.2, zorder=1)
        ax.annotate(f"mean drops below {floor_db:g} dB\nat {rc:.0f} km", (rc, 40),
                    color=INK2, fontsize=9, ha="center")

    ax.set_xlim(0, sc.range_max_m / 1000 * 1.02); ax.set_ylim(-20, 50)
    ax.set_xlabel("range / distance (km)"); ax.set_ylabel("received power over mean noise (dB)")
    leg = ax.legend(loc="upper right", frameon=False, fontsize=9)
    for t in leg.get_texts():
        t.set_color(INK2)
    dur = (a["scan_idx"].max() - a["scan_idx"].min()) * sc.scan_period_s / 60
    ax.set_title(f"Full-flight echo vs distance at a {floor_db:g} dB CFAR floor "
                 f"-- {AIRCRAFT} ({DATE})\n"
                 f"one aircraft, {len(a)} scans over {dur:.0f} min: echo fades with range",
                 color=INK)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150); plt.close(fig)


def plot_flight_track(sc, a, out_path, floor_db=8.0):
    """The aircraft's ground track on a PPI, relative to the radar. Points are
    coloured by whether the mean echo clears the CFAR floor at that range
    (inside vs beyond the detection horizon), so you see where along the real
    flight the radar loses it."""
    az = np.radians(a["true_azimuth_deg"].to_numpy())
    r = a["true_range_m"].to_numpy() / 1000
    e, n = r * np.sin(az), r * np.cos(az)
    snr_lin = 10 ** (a["snr_mean_db"].to_numpy() / 10)
    inside = 10 * np.log10(1 + snr_lin) >= floor_db
    horizon = sc.range_ref_m * (10 ** (sc.snr_ref_db / 10) / sc.threshold_lin(floor_db)) ** 0.25 / 1000

    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    rmax = sc.range_max_m / 1000
    for ring in (40, 80, 120, 160, 200):
        if ring <= rmax:
            ax.add_patch(plt.Circle((0, 0), ring, fill=False, color=GRID, lw=0.8, zorder=1))
            ax.annotate(f"{ring} km", (0, ring), color=MUTED, fontsize=8, ha="center", va="bottom")
    ax.add_patch(plt.Circle((0, 0), horizon, fill=False, color=INK, lw=1.2, ls=":", zorder=2))
    ax.annotate(f"{floor_db:g} dB detection horizon {horizon:.0f} km",
                (0, -horizon - 4), color=INK, fontsize=9, ha="center", va="top")

    ax.plot(e, n, color=GRID, lw=0.8, zorder=2)
    ax.scatter(e[inside], n[inside], s=14, color=C_TARGET, lw=0, zorder=4,
               label=f"echo ≥ {floor_db:g} dB (detectable)")
    ax.scatter(e[~inside], n[~inside], s=14, facecolor="none", edgecolor=MUTED, lw=0.7,
               zorder=4, label=f"echo < {floor_db:g} dB (below floor)")
    ax.plot(e[0], n[0], marker="o", color="#1baf7a", ms=11, zorder=6)
    ax.annotate("start", (e[0], n[0]), color=INK2, fontsize=9, ha="left", va="bottom")
    ax.plot(e[-1], n[-1], marker="s", color="#e34948", ms=10, zorder=6)
    ax.annotate("end", (e[-1], n[-1]), color=INK2, fontsize=9, ha="left", va="top")
    ax.plot(0, 0, marker="^", color=INK, ms=11, zorder=6)
    ax.annotate("radar", (0, -rmax * 0.04), color=INK2, fontsize=9, ha="center", va="top")

    lim = rmax * 1.1
    ax.set_aspect("equal"); ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("East (km)"); ax.set_ylabel("North (km)")
    ax.grid(False)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    leg = ax.legend(loc="upper left", frameon=False, fontsize=9, markerscale=1.4)
    for t in leg.get_texts():
        t.set_color(INK2)
    dur = (a["scan_idx"].max() - a["scan_idx"].min()) * sc.scan_period_s / 60
    ax.set_title(f"Flight track -- {AIRCRAFT} ({DATE})\n"
                 f"outbound {r.min():.0f} → {r.max():.0f} km over {dur:.0f} min; "
                 f"the radar loses it past the {horizon:.0f} km horizon", color=INK)
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

    for floor_db in FLOORS_DB:
        p = os.path.join(get_plot_dir(), f"stage05_ascope_{floor_db:g}db_distance.png")
        plot_full_flight(sc, a, p, floor_db)
        print(f"full-flight A-scope ({floor_db:g} dB) -> {p}")

    flight_path = os.path.join(get_plot_dir(), "stage05_flight.png")
    plot_flight_track(sc, a, flight_path)
    print(f"flight track -> {flight_path}")
    print(f"aircraft {AIRCRAFT}: {len(a)} scans, "
          f"{a.true_range_m.min()/1000:.0f}-{a.true_range_m.max()/1000:.0f} km")


if __name__ == "__main__":
    main()
