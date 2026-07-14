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
    rings = [r for r in (40, 80, 120, 160, 200) if r <= range_max_km]
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


def plot_bscope(dets: pd.DataFrame, k0: int, window_scans: int,
                range_max_km: float, title: str, out_path: str) -> None:
    """B-scope (range vs azimuth) of all detections in scans
    [k0, k0+window_scans) -- the radar's native measurement frame."""
    win = dets[(dets["scan_idx"] >= k0) & (dets["scan_idx"] < k0 + window_scans)]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for src, color, s, alpha, z in (("noise", C_NOISE, 1.5, 0.25, 2),
                                    ("clutter", C_CLUTTER, 6, 0.85, 4),
                                    ("target", C_TARGET, 5, 0.9, 5)):
        d = win[win["source"] == src]
        if d.empty:
            continue
        ax.scatter(d["azimuth_deg"], d["range_m"] / 1000, s=s, color=color,
                   alpha=alpha, lw=0, zorder=z, label=f"{src} ({len(d):,})")
    ax.set_xlim(0, 360); ax.set_ylim(0, range_max_km * 1.02)
    ax.set_xticks(range(0, 361, 45))
    ax.set_xlabel("azimuth (deg)"); ax.set_ylabel("range (km)")
    leg = ax.legend(loc="upper right", frameon=False, fontsize=9, markerscale=3)
    for t in leg.get_texts():
        t.set_color(INK2)
    ax.set_title(title, color=INK)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_rti(dets: pd.DataFrame, k0: int, window_scans: int, scan_t0: float,
             scan_period_s: float, range_max_km: float, title: str,
             out_path: str) -> None:
    """RTI (range vs time, azimuth collapsed) over scans [k0, k0+window_scans).

    The classic discrimination view: moving targets slope with their range
    rate, stationary clutter draws dead-flat horizontal lines, and noise
    speckles uniformly.
    """
    win = dets[(dets["scan_idx"] >= k0) & (dets["scan_idx"] < k0 + window_scans)]
    t_start = scan_t0 + k0 * scan_period_s
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for src, color, s, alpha, z in (("noise", C_NOISE, 1.5, 0.25, 2),
                                    ("clutter", C_CLUTTER, 4, 0.8, 4),
                                    ("target", C_TARGET, 4, 0.9, 5)):
        d = win[win["source"] == src]
        if d.empty:
            continue
        ax.scatter((d["t"] - t_start) / 60.0, d["range_m"] / 1000, s=s, color=color,
                   alpha=alpha, lw=0, zorder=z, label=f"{src} ({len(d):,})")
    ax.set_xlim(0, window_scans * scan_period_s / 60.0)
    ax.set_ylim(0, range_max_km * 1.02)
    ax.set_xlabel("time (min)"); ax.set_ylabel("range (km)")
    leg = ax.legend(loc="upper right", frameon=False, fontsize=9, markerscale=3)
    for t in leg.get_texts():
        t.set_color(INK2)
    ax.set_title(title, color=INK)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_ascope(sc, out_path: str, threshold_db: float = None) -> None:
    """A-scope: echo amplitude vs range along ONE beam, one scan --
    synthesized from the scenario's own statistics (Exp(1) noise per range
    cell). Two illustrative targets are shown at their MEAN echo power (a
    near one and a marginal far one) plus one clutter patch, with the CFAR
    floor at threshold_db (default sc.threshold_min_db) drawn. This is the
    picture in which 'lowering the CFAR threshold in dB' is defined.

    The noise realisation is seeded from sc.seed, so calling this at two
    thresholds yields identical noise -- only the floor line and the number
    of noise crossings above it differ, which is exactly the comparison.
    """
    threshold_db = sc.threshold_min_db if threshold_db is None else threshold_db
    rng = np.random.default_rng(sc.seed)
    n = int((sc.range_max_m - sc.range_min_m) / sc.range_resolution_m)
    r_km = (sc.range_min_m + sc.range_resolution_m * (np.arange(n) + 0.5)) / 1000
    amp = rng.exponential(1.0, n)                       # noise power, mean 1 (0 dB)

    # Mean-level echoes (no cherry-picked fluctuation draws): a strong near
    # target, a marginal far target, and a clutter patch.
    marks = []
    for label, rr_km, snr_lin in (
        ("target", 25.0, float(sc.snr_mean_lin(25_000.0))),
        ("target", 68.0, float(sc.snr_mean_lin(68_000.0))),
        ("clutter", 15.0, 10.0 ** (sc.clutter_snr_db / 10.0)),
    ):
        i = int(np.argmin(np.abs(r_km - rr_km)))
        amp[i] = 1.0 + snr_lin
        marks.append((label, r_km[i], 10 * np.log10(amp[i])))

    amp_db = np.maximum(10 * np.log10(amp), -20.0)      # clip deep noise nulls for display

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(r_km, amp_db, color=MUTED, lw=0.7, zorder=2)
    ax.axhline(threshold_db, color=INK, lw=1.4, ls="--", zorder=4)
    ax.annotate(f"CFAR floor {threshold_db:g} dB", (2, threshold_db + 0.7),
                color=INK, fontsize=9)
    ax.axhline(13.0, color=INK2, lw=1.2, ls=":", zorder=4)
    ax.annotate("conventional ~13 dB", (44, 13.7), color=INK2, fontsize=9)

    for label, rr, db in marks:
        color = C_TARGET if label == "target" else C_CLUTTER
        ax.plot([rr], [db], marker="o", ms=7, color=color, zorder=5)
        dx = -3.5 if label == "clutter" else 0.0     # keep the clutter label clear of the 13 dB line text
        ax.annotate(f"{label} {rr:.0f} km\n{db:.1f} dB", (rr + dx, db + 1.2),
                    color=color, fontsize=9, ha="center")

    fa = (amp_db >= threshold_db) & ~np.isin(
        np.arange(n), [int(np.argmin(np.abs(r_km - m[1]))) for m in marks])
    if fa.any():
        ax.plot(r_km[fa], amp_db[fa], "o", ms=5, color=C_NOISE, zorder=5,
                label=f"noise false alarms: {int(fa.sum())}")
        leg = ax.legend(loc="upper right", frameon=False, fontsize=9)
        for t in leg.get_texts():
            t.set_color(INK2)

    ax.set_xlim(0, sc.range_max_m / 1000 * 1.02); ax.set_ylim(-20, 30)
    ax.set_xlabel("range (km)"); ax.set_ylabel("received power over mean noise (dB)")
    ax.set_title(f"A-scope at a {threshold_db:g} dB CFAR floor -- one beam, one scan "
                 "(targets at mean echo power)\n"
                 "lowering the floor keeps more targets but admits more noise crossings",
                 color=INK)
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
