"""Radar scene (ground-truth PPI) for every day.

Reproduces the "radar scene" figure -- in-coverage GA trajectories (blue)
over the fixed ground-clutter map (orange) around the radar site -- for all
four days, and writes:

  * one PNG per day   (1_radar_scene_<date>.png)
  * one combined PNG  (1_radar_scene_all_days.png, 2x2 grid)

In-coverage trajectory paths come straight from the stage-6 beam crossings
(already range/elevation gated); clutter patches and the site come from the
frozen scenario.json. Style matches utils/plots.py.
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Make utils/ importable regardless of the caller's working directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.io import get_beam_crossings_dir, get_plot_dir, get_scenario_path

DATES = ["2022-06-06", "2022-06-13", "2022-06-20", "2022-06-27"]
# One distinct hue per day for the overlaid figure (orange is reserved for clutter).
DAY_COLORS = {
    "2022-06-06": "#2a78d6",   # blue
    "2022-06-13": "#2e8b57",   # green
    "2022-06-20": "#8a4fbd",   # purple
    "2022-06-27": "#d1495b",   # crimson
}
SITE_NAME = "Phoenix/Mesa AZ"

CROSSINGS_DIR = get_beam_crossings_dir()
PLOT_DIR = get_plot_dir()

# Palette / rc lifted from utils/plots.py so these match the rest of stage 5-9.
SURFACE = "#fcfcfb"; INK = "#0b0b0b"; INK2 = "#52514e"; MUTED = "#898781"
GRID = "#e1e0d9"; BASE = "#c3c2b7"
C_TARGET = "#2a78d6"; C_CLUTTER = "#eda100"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.edgecolor": BASE, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 12,
})


def _en_km(range_m, az_deg):
    """Compass azimuth (0=N, 90=E) polar -> East/North in km."""
    a = np.radians(az_deg)
    return range_m * np.sin(a) / 1000.0, range_m * np.cos(a) / 1000.0


def _ppi_frame(ax, range_max_km, label=True):
    """Range rings and radar marker on one PPI axis."""
    for r in (40, 80, 120, 160, 200):
        if r <= range_max_km:
            ax.add_patch(plt.Circle((0, 0), r, fill=False, color=GRID, lw=0.8, zorder=1))
            ax.annotate(f"{r} km", (0, r), color=MUTED, fontsize=8, ha="center", va="bottom",
                        zorder=2)
    lim = range_max_km * 1.1
    ax.set_aspect("equal")
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("East (km)"); ax.set_ylabel("North (km)")

    ax.plot(0, 0, marker="^", color=INK, ms=9, zorder=6)
    if label:
        ax.annotate("radar", (0, -range_max_km * 0.05), color=INK2, fontsize=9,
                    ha="center", va="top", zorder=7)


def load_day(date):
    """In-coverage trajectory paths (list of (E_km, N_km) arrays) and count."""
    cx = pd.read_csv(os.path.join(CROSSINGS_DIR, f"beam_crossings_{date}.csv"),
                     usecols=["scan_idx", "trajectory_id", "true_range_m", "true_azimuth_deg"])
    cx = cx.sort_values(["trajectory_id", "scan_idx"])
    e, n = _en_km(cx["true_range_m"].to_numpy(), cx["true_azimuth_deg"].to_numpy())
    cx = cx.assign(_e=e, _n=n)
    paths = [(g["_e"].to_numpy(), g["_n"].to_numpy()) for _, g in cx.groupby("trajectory_id")]
    return paths, cx["trajectory_id"].nunique()


def draw_scene(ax, date, paths, n_traj, range_max_km, color=C_TARGET,
               title_lines=True, frame=True):
    if frame:
        _ppi_frame(ax, range_max_km)
    for pe, pn in paths:
        ax.plot(pe, pn, color=color, lw=0.4, alpha=0.28, solid_capstyle="round", zorder=3)
    if title_lines:
        ax.set_title(f"Radar scene — {date} · {SITE_NAME}\n"
                     f"{n_traj:,} ground-truth GA trajectories in coverage (blue)", color=INK)


def main():
    with open(get_scenario_path()) as f:
        sc = json.load(f)
    range_max_km = sc["range_max_m"] / 1000.0

    os.makedirs(PLOT_DIR, exist_ok=True)
    days = {}
    for date in DATES:
        paths, n_traj = load_day(date)
        days[date] = (paths, n_traj)
        fig, ax = plt.subplots(figsize=(9, 9))
        draw_scene(ax, date, paths, n_traj, range_max_km)
        fig.tight_layout()
        out = os.path.join(PLOT_DIR, f"1_radar_scene_{date}.png")
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  {date}: {n_traj:,} trajectories -> {out}")

    # All four days overlaid on one PPI, coloured by day.
    fig, ax = plt.subplots(figsize=(10, 10))
    _ppi_frame(ax, range_max_km)
    for date in DATES:
        paths, n_traj = days[date]
        draw_scene(ax, date, paths, n_traj, range_max_km,
                   color=DAY_COLORS[date], title_lines=False, frame=False)
    handles = [plt.Line2D([], [], color=DAY_COLORS[d], lw=2.5,
                          label=f"{d}  ({days[d][1]:,})") for d in DATES]
    leg = ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=9,
                    title="trajectories in coverage", title_fontsize=9)
    leg.get_title().set_color(INK2)
    for t in leg.get_texts():
        t.set_color(INK2)
    ax.set_title(f"Radar scene — ground-truth GA trajectories in coverage · {SITE_NAME}\n"
                 "four survey days overlaid", color=INK)
    fig.tight_layout()
    out = os.path.join(PLOT_DIR, "1_radar_scene_all_days.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  overlaid -> {out}")


if __name__ == "__main__":
    main()
