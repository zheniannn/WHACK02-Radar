"""Shared plotting for stages 6-8 (PNG, light surface).

Colors follow the project's fixed entity mapping: targets blue, clutter
yellow (relieved by direct labels/legend), noise a deliberately recessive
gray. One window of scans is chosen deterministically from the beam
crossings so the stage 6/7/8 figures are directly comparable.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SURFACE = "#fcfcfb"; INK = "#0b0b0b"; INK2 = "#52514e"; MUTED = "#898781"
GRID = "#e1e0d9"; BASE = "#c3c2b7"
C_TARGET = "#2a78d6"; C_CLUTTER = "#eda100"; C_NOISE = "#898781"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.edgecolor": BASE, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 12,
})


def _en_km(range_m, az_deg):
    a = np.radians(az_deg)
    return range_m * np.sin(a) / 1000.0, range_m * np.cos(a) / 1000.0


def _ppi_axes(ax, range_max_km):
    rings = [r for r in (20, 40, 60, 80, 100) if r <= range_max_km]
    for r in rings:
        ax.add_patch(plt.Circle((0, 0), r, fill=False, color=GRID, lw=0.8, zorder=1))
        ax.annotate(f"{r} km", (0, r), color=MUTED, fontsize=8, ha="center", va="bottom")
    lim = range_max_km * 1.1
    ax.set_aspect("equal"); ax.grid(False)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("East (km)"); ax.set_ylabel("North (km)")
    ax.plot(0, 0, marker="^", color=INK, ms=9, zorder=6)
    ax.annotate("radar", (0, -range_max_km * 0.05), color=INK2, fontsize=9, ha="center", va="top")


def densest_window(crossings_path: str, window_scans: int = 90) -> int:
    """First scan index of the busiest window of beam crossings -- computed
    from stage 6's deterministic geometry so every stage plots the same
    window and the figures are comparable."""
    cx = pd.read_csv(crossings_path, usecols=["scan_idx"])
    counts = cx["scan_idx"].value_counts().sort_index()
    full = counts.reindex(range(counts.index.max() + 1), fill_value=0).to_numpy()
    return int(np.argmax(np.convolve(full, np.ones(window_scans, int), "valid")))


def plot_detection_window(dets: pd.DataFrame, k0: int, window_scans: int,
                          range_max_km: float, title: str, out_path: str,
                          horizon_km: float = None) -> None:
    """PPI scatter of all detections in scans [k0, k0+window_scans).
    horizon_km draws a dotted ring (e.g. a deterministic detection horizon)."""
    win = dets[(dets["scan_idx"] >= k0) & (dets["scan_idx"] < k0 + window_scans)]
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    _ppi_axes(ax, range_max_km)
    if horizon_km is not None:
        ax.add_patch(plt.Circle((0, 0), horizon_km, fill=False, color=INK,
                                lw=1.2, ls=":", zorder=3))
        ax.annotate(f"detection horizon {horizon_km:.0f} km",
                    (0, -horizon_km - 2), color=INK, fontsize=9, ha="center", va="top")
    for src, color, s, alpha, z in (("noise", C_NOISE, 1.5, 0.25, 2),
                                    ("clutter", C_CLUTTER, 5, 0.8, 4),
                                    ("target", C_TARGET, 5, 0.9, 5)):
        d = win[win["source"] == src]
        if d.empty:
            continue
        e, n = _en_km(d["range_m"].to_numpy(), d["azimuth_deg"].to_numpy())
        ax.scatter(e, n, s=s, color=color, alpha=alpha, lw=0, zorder=z,
                   label=f"{src} ({len(d):,})")
    leg = ax.legend(loc="upper left", frameon=False, fontsize=9, markerscale=3)
    for t in leg.get_texts():
        t.set_color(INK2)
    ax.set_title(title, color=INK)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def longest_miss_run(detected: np.ndarray) -> int:
    """Length of the longest run of consecutive False values."""
    x = (~detected.astype(bool)).astype(int)
    d = np.diff(np.concatenate(([0], x, [0])))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    return int((ends - starts).max()) if starts.size else 0


def per_track_drop_table(truth_all: pd.DataFrame, min_crossings: int = 30,
                         gap_scans: int = 3) -> pd.DataFrame:
    """One row per trajectory: median range and whether it contains a miss
    gap of >= gap_scans consecutive scans (a simple 'track dropped' proxy)."""
    rows = []
    for tid, g in truth_all.sort_values(["trajectory_id", "scan_idx"]).groupby("trajectory_id"):
        det = g["detected"].to_numpy()
        if len(det) < min_crossings:
            continue
        rows.append((tid, float(g["true_range_m"].median()),
                     longest_miss_run(det) >= gap_scans))
    return pd.DataFrame(rows, columns=["trajectory_id", "r_median_m", "dropped"])


def plot_max_range(truth_all: pd.DataFrame, track_table: pd.DataFrame, sc,
                   r50_emp_km: float, drop50_km: float, gap_scans: int,
                   out_path: str) -> None:
    """Stage 9's headline figure: Pd vs range and track-drop fraction vs
    range, with the two derived maximum-range markers."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7.5), sharex=True)
    edges = np.linspace(sc.range_min_m, sc.range_max_m, 17)

    # Pd vs range: empirical + closed form.
    mid = (edges[:-1] + edges[1:]) / 2
    emp = [truth_all.loc[truth_all["true_range_m"].between(lo, hi), "detected"].mean()
           for lo, hi in zip(edges[:-1], edges[1:])]
    r_th = np.linspace(sc.range_min_m, sc.range_max_m, 300)
    ax1.plot(r_th / 1000, sc.pd(r_th), color=INK2, lw=1.5, ls="--", label="Swerling-1 theory")
    ax1.plot(mid / 1000, emp, color=C_TARGET, lw=2, marker="o", ms=4, label="empirical")
    ax1.axhline(0.5, color=GRID, lw=1)
    ax1.axvline(r50_emp_km, color=INK, lw=1.2, ls=":")
    ax1.annotate(f"Pd = 0.5 at {r50_emp_km:.0f} km", (r50_emp_km - 1, 0.53),
                 color=INK, fontsize=9, ha="right")
    ax1.set_ylabel("probability of detection"); ax1.set_ylim(0, 1.03)
    leg = ax1.legend(frameon=False, fontsize=9, loc="lower left")
    for t in leg.get_texts():
        t.set_color(INK2)
    ax1.set_title("Stage 9 — maximum range before the radar drops a trajectory", color=INK)

    # Track-drop fraction vs range.
    mids, fracs = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = track_table[(track_table["r_median_m"] >= lo) & (track_table["r_median_m"] < hi)]
        if len(sel) >= 5:
            mids.append((lo + hi) / 2000)
            fracs.append(sel["dropped"].mean())
    ax2.plot(mids, fracs, color=C_TARGET, lw=2, marker="o", ms=4)
    ax2.axhline(0.5, color=GRID, lw=1)
    ax2.axvline(drop50_km, color=INK, lw=1.2, ls=":")
    ax2.annotate(f"50% of tracks broken at {drop50_km:.0f} km", (drop50_km - 1, 0.53),
                 color=INK, fontsize=9, ha="right")
    ax2.set_xlabel("range (km)")
    ax2.set_ylabel(f"fraction of tracks with a >={gap_scans}-scan gap")
    ax2.set_xlim(0, sc.range_max_m / 1000 * 1.02); ax2.set_ylim(0, 1.03)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
