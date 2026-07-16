"""Stage 5: choose the radar site from ground-truth traffic density, freeze
the full radar scenario (coverage, accuracy, SNR model, CFAR floor, clutter
map) into scenario.json, and produce the data-derived detection figures for
one real flight.

The detection figures use a real 2022-06-06 flight -- N118AT, a Piper
PA-44-180 Seminole (icao24 a049fd), outbound 8 -> 200 km -- together with
the scenario physics, so they need the beam-crossing geometry. Building it
here also warms the deterministic cache that stages 6-9 share.

Outputs:
  scenario.json
  3_ascope_8db_distance.png / 3_ascope_0db_distance.png
      echo power vs range across the whole flight (mean radar-equation curve
      + per-scan Swerling draws, detected/missed) against the Exp(1) noise
      floor, at each CFAR floor.
  3_flight_ppi.png
      the aircraft's ground track on a PPI, blue inside the detection
      horizon and grey beyond.

Usage:
    python scripts/05_radar_scenario.py
    python scripts/05_radar_scenario.py --range-max-km 60 --threshold-min-db 10
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Make utils/ importable regardless of the caller's working directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.beam_crossings import discover_input_files, ensure_beam_crossings
from utils.io import (
    get_beam_crossings_dir, get_plot_dir, get_scenario_path, get_trajectories_dir,
)
from utils.plots import C_NOISE, C_TARGET, GRID, INK, INK2, MUTED
from utils.scenario import C_M_S, Scenario, generate_clutter_patches, select_site

# Reference flight for the detection figures.
FLIGHT_DATE = "2022-06-06"
TRAJECTORY_ID = "a049fd_1654554529_r0"
AIRCRAFT = "N118AT  Piper PA-44-180 Seminole"
FLOORS_DB = (8.0, 0.0)

# Calibration target: the operating (post-integration) mean SNR at range_ref.
# PRF and the scan geometry set the integration gain; the single-pulse SNR is
# then whatever yields this operating anchor (so the anchor stays calibrated
# but its integration component is derived from physical parameters).
OPERATING_SNR_REF_DB = 15.0


def parse_args():
    parser = argparse.ArgumentParser(description="Select radar site and write scenario.json.")
    parser.add_argument("--input-dir", type=str, default=None,
                        help="Directory of stage-4 trajectory CSVs (default: active/trajectories_10s).")
    parser.add_argument("--output", type=str, default=None,
                        help="Scenario JSON path (default: active/radar/scenario.json).")
    parser.add_argument("--range-max-km", type=float, default=200.0,
                        help="Instrumented range in km (default: 200).")
    parser.add_argument("--threshold-min-db", type=float, default=8.0,
                        help="CFAR floor in dB; measurements are recorded down to this (default: 8).")
    parser.add_argument("--seed", type=int, default=20220606,
                        help="RNG seed frozen into the scenario (default: 20220606).")
    parser.add_argument("--no-figures", action="store_true",
                        help="Skip the detection figures (and the beam-crossing build they need).")
    return parser.parse_args()


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
                 f"-- {AIRCRAFT} ({FLIGHT_DATE})\n"
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
    # SNR >= floor, matching the deterministic horizon ring below (and stage 8).
    inside = 10 * np.log10(snr_lin) >= floor_db
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
    ax.set_title(f"Flight track -- {AIRCRAFT} ({FLIGHT_DATE})\n"
                 f"outbound {r.min():.0f} → {r.max():.0f} km over {dur:.0f} min; "
                 f"the radar loses it past the {horizon:.0f} km horizon", color=INK)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150); plt.close(fig)


def make_detection_figures(sc, input_dir):
    """Build the beam-crossing cache and render the real-flight figures."""
    print("\nbuilding beam-crossing cache for the detection figures...")
    ensure_beam_crossings(input_dir, get_beam_crossings_dir(), sc)
    cx_path = os.path.join(get_beam_crossings_dir(), f"beam_crossings_{FLIGHT_DATE}.csv")
    if not os.path.exists(cx_path):
        print(f"(no crossings for {FLIGHT_DATE}; skipping detection figures)")
        return
    a = pd.read_csv(cx_path)
    a = a[a["trajectory_id"] == TRAJECTORY_ID].sort_values("scan_idx")
    if a.empty:
        print(f"(reference flight {TRAJECTORY_ID} not in coverage; skipping figures)")
        return
    for floor_db in FLOORS_DB:
        plot_full_flight(sc, a, os.path.join(get_plot_dir(), f"3_ascope_{floor_db:g}db_distance.png"),
                         floor_db)
    plot_flight_track(sc, a, os.path.join(get_plot_dir(), "3_flight_ppi.png"))
    print(f"detection figures ({AIRCRAFT}, {len(a)} scans "
          f"{a.true_range_m.min()/1000:.0f}-{a.true_range_m.max()/1000:.0f} km) -> {get_plot_dir()}")


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir or get_trajectories_dir()
    output_path = args.output or get_scenario_path()

    day_files = discover_input_files(input_dir)
    if not day_files:
        raise FileNotFoundError(f"No stage-4 trajectory CSVs found in {input_dir}")
    print(f"Selecting site from {len(day_files)} day(s) of trajectories...")

    site = select_site([path for _, path in day_files])
    sc = Scenario(
        site_lat_deg=site["site_lat_deg"],
        site_lon_deg=site["site_lon_deg"],
        site_alt_m=site["site_alt_m"],
        range_max_m=args.range_max_km * 1000.0,
        threshold_min_db=args.threshold_min_db,
        seed=args.seed,
    )
    # Waveform: set the PRF so range_max is unambiguous (R_u = c / 2PRF).
    sc.prf_hz = C_M_S / (2.0 * sc.range_max_m)
    # S-band link budget: solve the system loss that makes the radar equation
    # reproduce the calibrated operating anchor after integration. Single-pulse
    # SNR is inversely proportional to loss, so in dB the required loss is the
    # loss-free single-pulse SNR minus the target single-pulse SNR.
    target_pulse_db = OPERATING_SNR_REF_DB - sc.integration_gain_db()
    sc.system_loss_db = 0.0
    sc.system_loss_db = sc.snr_pulse_ref_db - target_pulse_db

    sc.clutter_patches = generate_clutter_patches(sc, np.random.default_rng(sc.seed))
    sc.save(output_path)

    print(f"site: lat {sc.site_lat_deg}, lon {sc.site_lon_deg}, alt {sc.site_alt_m} m "
          f"(densest cell: {site['density_samples']} samples ~ "
          f"{site['density_samples'] * 10 / 3600:.0f} flight-hours)")
    print(f"coverage: {sc.range_min_m / 1000:.0f}-{sc.range_max_m / 1000:.0f} km, "
          f"elevation {sc.elevation_min_deg}-{sc.elevation_max_deg} deg, scan {sc.scan_period_s}s")
    print(f"cells/scan: {sc.n_cells()}, CFAR floor: {sc.threshold_min_db} dB "
          f"(expected false alarms/scan: {sc.expected_false_alarms_per_scan():.1f})")
    print(f"clutter patches: {len(sc.clutter_patches)}")

    print(f"\n{sc.band()} link budget:")
    print(f"  f {sc.frequency_hz / 1e9:.2f} GHz (lambda {sc.wavelength_m() * 100:.1f} cm), "
          f"Pt {sc.tx_peak_power_w / 1e3:.0f} kW, G {sc.antenna_gain_db:.0f} dBi")
    print(f"  B {sc.bandwidth_hz() / 1e6:.1f} MHz (from {sc.range_resolution_m:.0f} m resolution), "
          f"NF {sc.noise_figure_db:.0f} dB, system loss {sc.system_loss_db:.1f} dB (solved)")
    print(f"  PRF {sc.prf_hz:.0f} Hz (unambiguous {sc.unambiguous_range_m() / 1000:.0f} km = range_max), "
          f"dwell {sc.dwell_time_s() * 1000:.1f} ms -> {sc.pulses_per_dwell():.1f} pulses")
    print(f"  single-pulse {sc.snr_pulse_ref_db:.2f} dB + integration {sc.integration_gain_db():.2f} dB "
          f"= {sc.snr_ref_db:.2f} dB operating @ {sc.range_ref_m / 1000:g} km")

    # Radar equation, evaluated at the scenario's calibration (see Scenario.snr_mean_lin).
    print(f"\nradar equation ({sc.rcs_ref_m2:g} m^2 target, "
          f"{sc.snr_ref_db:g} dB @ {sc.range_ref_m / 1000:g} km, R^-4):")
    print(f"  {'range':>8} | {'mean SNR':>9} | {'Pd @ floor':>10}")
    for r_km in (40, 80, 120, 160, 200):
        snr_db = 10 * np.log10(sc.snr_mean_lin(r_km * 1000.0))
        print(f"  {r_km:>5} km | {snr_db:>6.1f} dB | {float(sc.pd(r_km * 1000.0)):>10.3f}")

    print(f"\nscenario written to: {os.path.abspath(output_path)}")

    if not args.no_figures:
        make_detection_figures(sc, input_dir)


if __name__ == "__main__":
    main()
