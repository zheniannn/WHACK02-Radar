# WHACK02-Radar

Three-stage 2D radar measurement simulator. Consumes the ground-truth
conventional-GA trajectories produced by
[WHACK01-Preprocessing](https://github.com/zheniannn/WHACK01-Preprocessing)
and produces per-scan radar detections — genuine targets, noise false
alarms, and persistent ground clutter — for studying track-based
target-vs-clutter discrimination at low CFAR thresholds.

The pipeline separates **deterministic** from **stochastic**: stage 6
computes pure geometry once (the slow part); stage 7 draws the random
measurement layer on top and can be re-run cheaply with different seeds
(Monte Carlo) or noise settings.

## Structure

```
WHACK02-Radar/
├── requirements.txt
├── scripts/
│   ├── 05_radar_scenario.py          # stage 5: site selection -> scenario.json
│   ├── 06_beam_crossings.py          # stage 6: trajectories -> deterministic radar truth
│   └── 07_generate_measurements.py   # stage 7: truth -> detections (noise, FAs, clutter)
└── utils/
    ├── io.py                          # input/output path resolution
    ├── geometry.py                    # geodetic -> ENU -> range/azimuth/elevation
    ├── scenario.py                    # stage 5 rules: site, radar physics, scenario schema
    ├── beam_crossings.py              # stage 6 rules: beam timing, coverage gating
    └── measurements.py                # stage 7 rules: detection draws, noise, FAs, clutter
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
    ├── beam_crossings/    # stage 6 output
    └── measurements/      # stage 7 output
```

## Usage

```bash
python scripts/05_radar_scenario.py
python scripts/06_beam_crossings.py
python scripts/07_generate_measurements.py
```

---

## Stage 5 — `05_radar_scenario.py`

Chooses the radar site and freezes every simulation parameter into
`scenario.json` so later stages are reproducible.

- **Site**: centre of the densest 0.25° traffic cell across all days
  (density = sample count = dwell time, so busy training areas outweigh
  fast transits). Site elevation is estimated from the 1st percentile of
  nearby flight altitudes minus 150 m (terrain proxy — no DEM is used).
- **Radar model** (defaults; edit `scenario.json` or use CLI flags):
  2D fan-beam surveillance radar, 10 s scan period, 1–80 km instrumented
  range, 0.3–30° elevation fan, 150 m range resolution, 1.5° beamwidth,
  σ_range = 50 m, σ_azimuth = 0.2°.
- **Radar physics live on the `Scenario` class** (`utils/scenario.py`):
  the calibrated radar equation (mean SNR 15 dB for a 1 m² target at
  50 km, R⁻⁴ two-way falloff), `Pfa(τ) = exp(−τ)`, and the Swerling-1
  `Pd = Pfa^(1/(1+SNR))`. Stages 6–7 only *apply* this model; the script
  prints an SNR/Pd-vs-range table at scenario time.
- **Clutter map**: 25 stationary patches, uniform in azimuth, within
  40 km, mean SNR 12 dB — fixed across days (ground clutter is static).

CLI flags: `--range-max-km`, `--threshold-min-db`, `--seed`,
`--input-dir`, `--output`.

## Stage 6 — `06_beam_crossings.py` (deterministic truth)

Computes, per day, **when and where the radar's beam crosses each
trajectory** — with zero randomness.

1. **Scan epochs** — a fixed 10 s grid covering the day.
2. **Beam-crossing times** — a rotating beam hits a target at
   `scan_start + azimuth/360 × T`; the azimuth depends on where the target
   is at that time, solved with two fixed-point iterations (GA targets
   move < 1 km per scan, so convergence error is far below measurement
   noise). Ground truth is interpolated to the crossing time; never
   extrapolated beyond the trajectory's span.
3. **Coverage gating** — slant range within [range_min, range_max] and
   elevation within the fan. Altitude is used only for slant-range and
   elevation computation (2D radar: it is not measured).
4. **Mean SNR** — the scenario's radar equation evaluated at the true range.

**Output:** `beam_crossings_<date>.csv` — one row per in-coverage crossing
(`scan_idx, t, trajectory_id, icao24, true_range_m, true_azimuth_deg,
true_elevation_deg, snr_mean_db`) — plus `beam_crossings_summary.csv`,
which also carries each day's scan grid (`scan_t0`, `n_scans`) for stage 7.

**Validation gate:** every crossing inside the coverage gates; `snr_mean_db`
exactly equals the radar equation at the true range; crossing cadence sits
on the scan grid.

## Stage 7 — `07_generate_measurements.py` (stochastic layer)

Draws the random measurement process on top of stage 6's truth.

1. **Detection draws** — square-law detector, exponential noise,
   Swerling 1 target fluctuation: each crossing draws a measured power
   `z ~ Exp(1 + SNR_mean)`; a measurement is recorded when
   `z ≥ threshold_min`. `snr_db = 10·log₁₀(z)` is stored, so **any CFAR
   threshold ≥ the floor can be applied post-hoc** by filtering on
   `snr_db` — one dataset supports a full ROC sweep.
2. **Measurement noise** — Gaussian on range and azimuth (wrapped to
   [0, 360)) for detected crossings.
3. **False alarms** — Poisson over resolution cells × scans at
   `Pfa(floor)`; conditional power uses the memoryless property
   (`z = τ + Exp(1)`). Uniform over range and azimuth.
4. **Clutter** — each patch fluctuates per scan around its mean SNR and
   is measured with the same noise model; labelled `clutter`.

**Outputs (per day):**

- `radar_truth_<date>.csv` — every beam crossing with its measured SNR and
  detection outcome. This is the denominator for Pd and
  track-completeness metrics.
- `radar_detections_<date>.csv` — what a tracker sees: `scan_idx`, `t`,
  `range_m`, `azimuth_deg`, `snr_db`, `source`
  (`target` / `noise` / `clutter`), plus truth linkage
  (`trajectory_id`, `icao24`, true position) for evaluation only.
- `radar_measurements_summary.csv` — one row per day.

**Validation gate:** empirical false-alarm rate within 5σ of
`n_cells × Pfa(floor)`; empirical Pd tracks the Swerling-1 closed form
(±0.05) in three range bins; target measurement residuals reproduce
σ_range and σ_azimuth within 5%.

**Monte Carlo:** `--seed N --output-dir <dir>` produces an independent
measurement realisation of the same geometry without recomputing stage 6.

## Extending

Stage-5 rules and all radar physics live in `utils/scenario.py`; stage-6
geometry in `utils/beam_crossings.py`; stage-7 stochastics in
`utils/measurements.py`. The scenario JSON is the single source of truth
for parameters — edit it (or rerun stage 5 with flags) and rerun stages
6–7.
