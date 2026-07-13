# WHACK02-Radar

Two-stage 2D radar measurement simulator. Consumes the ground-truth
conventional-GA trajectories produced by
[WHACK01-Preprocessing](https://github.com/zheniannn/WHACK01-Preprocessing)
and produces per-scan radar detections — genuine targets, noise false
alarms, and persistent ground clutter — for studying track-based
target-vs-clutter discrimination at low CFAR thresholds.

## Structure

```
WHACK02-Radar/
├── requirements.txt
├── scripts/
│   ├── 05_radar_scenario.py          # stage 5: site selection -> scenario.json
│   └── 06_generate_measurements.py   # stage 6: trajectories -> radar detections
└── utils/
    ├── io.py                          # input/output path resolution
    ├── geometry.py                    # geodetic -> ENU -> range/azimuth/elevation
    ├── scenario.py                    # stage 5 rules: density, site, scenario schema
    └── radar_sim.py                   # stage 6 rules: beam timing, SNR, detection draws
```

## Requirements

Python ≥ 3.10 with `pandas` and `numpy`:

```bash
pip install -r requirements.txt
```

## Data layout

Same convention as WHACK01: the data root defaults to `data/` next to the
repository (override with `WHACK_DATA_ROOT`).

```
<data root>/active/
├── trajectories_10s/      # WHACK01 stage 4 output (this repo's input)
└── radar/
    ├── scenario.json      # stage 5 output
    └── measurements/      # stage 6 output
```

## Usage

```bash
python scripts/05_radar_scenario.py
python scripts/06_generate_measurements.py
```

---

## Stage 5 — `05_radar_scenario.py`

Chooses the radar site and freezes every simulation parameter into
`scenario.json` so stage-6 runs are reproducible.

- **Site**: centre of the densest 0.25° traffic cell across all days
  (density = sample count = dwell time, so busy training areas outweigh
  fast transits). Site elevation is estimated from the 1st percentile of
  nearby flight altitudes minus 150 m (terrain proxy — no DEM is used).
- **Radar model** (defaults; edit `scenario.json` or use CLI flags):
  2D fan-beam surveillance radar, 10 s scan period, 1–80 km instrumented
  range, 0.3–30° elevation fan, 150 m range resolution, 1.5° beamwidth,
  σ_range = 50 m, σ_azimuth = 0.2°.
- **SNR model**: radar equation with R⁻⁴ falloff; mean SNR 15 dB for a
  1 m² target at 50 km. All aircraft use RCS = 1 m² (Swerling 1
  fluctuation covers scan-to-scan variation).
- **Clutter map**: 25 stationary patches, uniform in azimuth, within
  40 km, mean SNR 12 dB — fixed across days (ground clutter is static).

CLI flags: `--range-max-km`, `--threshold-min-db`, `--seed`,
`--input-dir`, `--output`.

## Stage 6 — `06_generate_measurements.py`

Generates per-day radar measurements from the trajectories per the frozen
scenario.

### Method

1. **Scan epochs** — a fixed 10 s grid covering the day.
2. **Beam-crossing times** — a rotating beam hits a target at
   `scan_start + azimuth/360 × T`; the azimuth depends on where the target
   is at that time, solved with two fixed-point iterations (GA targets
   move < 1 km per scan, so convergence error is far below measurement
   noise). Ground truth is interpolated to the crossing time; no
   extrapolation beyond the trajectory's span.
3. **Coverage gating** — slant range within [range_min, range_max] and
   elevation within the fan. Altitude is used only for slant-range and
   elevation computation (2D radar: it is not measured).
4. **Detection draws** — square-law detector, exponential noise,
   Swerling 1 target fluctuation: `Pfa(τ) = exp(−τ)`,
   `Pd(τ, s̄) = Pfa^(1/(1+s̄))` (linear units). Each opportunity draws a
   measured power `z ~ Exp(1 + s̄)`; a measurement is recorded when
   `z ≥ threshold_min`. `snr_db = 10·log₁₀(z)` is stored, so **any CFAR
   threshold ≥ the floor can be applied post-hoc** by filtering on
   `snr_db` — one dataset supports a full ROC sweep.
5. **Measurement noise** — Gaussian on range and azimuth (wrapped to
   [0, 360)).
6. **False alarms** — Poisson over resolution cells × scans at
   `Pfa(floor)`; conditional power uses the memoryless property
   (`z = τ + Exp(1)`). Uniform over range and azimuth.
7. **Clutter** — each patch fluctuates per scan around its mean SNR and
   is measured with the same noise model; labelled `clutter`.

### Outputs (per day)

- `radar_truth_<date>.csv` — every in-coverage beam crossing of every
  trajectory (detected or not): true range/azimuth/elevation, mean and
  measured SNR, detection flag. This is the denominator for Pd and
  track-completeness metrics.
- `radar_detections_<date>.csv` — what a tracker sees: `scan_idx`, `t`,
  `range_m`, `azimuth_deg`, `snr_db`, `source`
  (`target` / `noise` / `clutter`), plus truth linkage
  (`trajectory_id`, `icao24`, true position) for evaluation only.
- `radar_measurements_summary.csv` — one row per day.

A **validation gate** runs after processing and raises on failure: all
opportunities within the coverage gates; empirical false-alarm rate within
5σ of `n_cells × Pfa(floor)`; empirical Pd tracks the Swerling-1 closed
form (±0.05) in three range bins; target measurement residuals reproduce
σ_range and σ_azimuth within 5%.

## Extending

Stage-5 rules (density binning, terrain proxy, defaults) live in
`utils/scenario.py`; stage-6 physics (beam timing, SNR, detection and
clutter draws) in `utils/radar_sim.py`. The scenario JSON is the single
source of truth for parameters — edit it (or rerun stage 5 with flags) and
rerun stage 6.
