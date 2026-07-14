# WHACK02-Radar

2D radar measurement simulator built on the ground-truth conventional-GA
trajectories from
[WHACK01-Preprocessing](https://github.com/zheniannn/WHACK01-Preprocessing).
Stages 6–9 form a progressive-complexity ladder for studying track-based
target-vs-clutter discrimination at low CFAR thresholds: stages 6/8 are
clean (fixed vs radar-equation SNR), stages 7/9 add the contamination, so
every effect in the figures has a single cause.

| Stage | SNR | Fluctuation | Meas. noise | False alarms | Clutter |
|---|---|---|---|---|---|
| 6 | fixed 15 dB | — | — | — | — |
| 7 | fixed 15 dB | ✓ | ✓ | ✓ | ✓ |
| 8 | radar equation (R⁻⁴) | — | — | — | — |
| 9 | radar equation (R⁻⁴) | ✓ | ✓ | ✓ | ✓ |

## Structure

```
WHACK02-Radar/
├── requirements.txt
├── scripts/
│   ├── 05_radar_scenario.py          # stage 5: site + radar definition -> scenario.json
│   ├── 06_trajectories_clean.py      # stage 6: trajectories only, fixed SNR, no clutter/noise
│   ├── 07_trajectories_cluttered.py  # stage 7: fixed SNR + clutter + noise
│   ├── 08_trajectories_radar_equation.py # stage 8: radar-equation SNR, no clutter/noise
│   └── 09_radar_equation_cluttered.py    # stage 9: radar-equation SNR + clutter + noise + max range
└── utils/
    ├── io.py                          # input/output path resolution
    ├── geometry.py                    # geodetic -> ENU -> range/azimuth/elevation
    ├── scenario.py                    # stage 5 rules: site, radar physics, scenario schema
    ├── beam_crossings.py              # shared deterministic geometry (cached for stages 6-9)
    ├── measurements.py                # MeasurementConfig + the stochastic measurement layer
    └── plots.py                       # shared PPI / max-range figures
```

## Requirements

Python ≥ 3.10:

```bash
pip install -r requirements.txt
```

## Data layout

Same convention as WHACK01: the data root defaults to `data/` next to the
repository (override with `WHACK_DATA_ROOT`).

```
<data root>/
├── plot/                      # figures from all stages
└── active/
    ├── trajectories_10s/      # WHACK01 stage 4 output (this repo's input)
    └── radar/
        ├── scenario.json      # stage 5 output
        ├── beam_crossings/    # deterministic geometry cache (shared by 6-9)
        ├── stage06/           # per-day truth + detections per stage
        ├── stage07/
        ├── stage08/
        └── stage09/           #   ... + max_range_report.json
```

## Usage

```bash
python scripts/05_radar_scenario.py
python scripts/06_trajectories_clean.py
python scripts/07_trajectories_cluttered.py
python scripts/08_trajectories_radar_equation.py
python scripts/09_radar_equation_cluttered.py
```

Whichever of stages 6–9 runs first computes the beam-crossing cache; the
others reuse it (a fingerprint sidecar recomputes it automatically if the
scenario's geometry changes).

---

## Stage 5 — `05_radar_scenario.py`

Defines the radar: location, settings, characteristics. Simulates nothing.

- **Site**: centre of the densest 0.25° traffic cell across all days
  (density = sample count = dwell time) → Phoenix/Mesa, AZ. Site elevation
  from the 1st percentile of nearby flight altitudes minus 150 m (terrain
  proxy — no DEM).
- **Radar model**: 2D fan-beam surveillance radar, 10 s scan, 1–80 km,
  0.3–30° elevation fan, 150 m × 1.5° resolution cells, σ_range = 50 m,
  σ_azimuth = 0.2° (≈ beamwidth/7.5, a standard monopulse fraction).
- **Radar physics live on the `Scenario` class**: calibrated radar
  equation (15 dB for 1 m² at 50 km, R⁻⁴), `Pfa(τ) = exp(−τ)`, Swerling-1
  `Pd = Pfa^(1/(1+SNR))`. Later stages only apply this model.
- **Clutter map**: 25 stationary patches (positions frozen here — ground
  clutter doesn't move between days or between Monte-Carlo runs), mean
  SNR 12 dB, within 40 km.
- Everything is frozen into `scenario.json` with the RNG seed; all later
  stages are reproducible functions of that file.
- The **A-scope** figures are data-derived (see
  `scripts/05b_real_ascope.py`, run after stage 6): a real 2022-06-06 flight,
  N118AT (Piper PA-44-180 Seminole), shown at two scans of its own outbound
  track -- strong at 25 km (27 dB) and, 14 min later, marginal at 68 km
  (10 dB), between the 8 dB floor and the conventional 13 dB. Generated at
  both an 8 dB and a 5 dB CFAR floor (`stage05_ascope_8db.png`,
  `stage05_ascope_5db.png`) so the effect of lowering the floor is visible.

CLI: `--range-max-km`, `--threshold-min-db`, `--seed`, `--input-dir`, `--output`.

## Shared geometry — the beam-crossing cache

A rotating beam hits a target at `scan_start + azimuth/360 × T`, where the
azimuth itself depends on the target's position at that time (solved with
two fixed-point iterations; GA targets move < 1 km per scan). Ground truth
is interpolated to the crossing instant, never extrapolated. Coverage
gating: slant range in [1, 80] km, elevation in the fan (altitude is used
only for slant range and elevation — a 2D radar does not measure it). This
is fully deterministic, so it's computed once into
`radar/beam_crossings/` and shared by stages 6–9.

## Stage 6 — `06_trajectories_clean.py`

**Aircraft trajectories only — no clutter, no noise — at a fixed SNR of
15 dB.** Fluctuation, measurement noise, false alarms, and clutter are all
off; 15 dB always clears the 8 dB floor, so every in-coverage beam
crossing is recorded exactly. The output is the pure per-scan radar view
of the trajectories, and the PPI figure shows clean tracks.

Gate: every crossing detected; targets only; measurements identical to
truth; `snr_db` exactly 15.

## Stage 7 — `07_trajectories_cluttered.py`

**Same fixed 15 dB SNR, now with clutter and noise.** Swerling-1
fluctuation (Pd ≈ 0.82 at the 8 dB floor — *uniform at every range*),
Gaussian measurement noise, Poisson false alarms over the 126,240 CFAR
cells (~230/scan at 8 dB), and the 25 persistent clutter patches. The
figure uses the same 15-minute window as stage 6, so the added
contamination is the only difference.

Gate: Pd uniform across range bins and equal to the fixed-SNR closed form;
false-alarm rate within 5σ of theory; measurement σs reproduced.

`--seed` re-rolls the stochastic layer (Monte Carlo) without recomputing
geometry.

## Stage 8 — `08_trajectories_radar_equation.py`

**Radar-equation SNR, no clutter or noise.** With fluctuation and noise
off, detection is deterministic: a crossing is recorded iff its mean SNR
clears the 8 dB floor, i.e. iff its range is within the closed-form
detection horizon `R = range_ref · (snr_ref/τ)^(1/4)` ≈ **74.8 km**. The
figure shows clean tracks ending exactly at that ring — the radar
equation's range limit isolated from every stochastic effect.

Gate: detection ≡ (range ≤ horizon) exactly; targets only; measurements
identical to truth; `snr_db` equals the radar equation.

## Stage 9 — `09_radar_equation_cluttered.py`

**Full physics: SNR from the radar equation, with clutter and noise.**
Distant targets fade under the R⁻⁴ law (Pd ≈ 1.0 near, ~0.34 at 80 km),
and the stage answers: *what is the maximum range before the radar drops
an aircraft trajectory?* Two limits are computed and written to
`max_range_report.json`:

- **Detection limit** — the range where single-scan Pd falls to 0.5
  (empirical, checked against the closed form).
- **Tracking limit** — the range where 50% of tracks contain a gap of ≥ 3
  consecutive missed scans (a simple track-drop proxy: most trackers coast
  only a couple of scans before deleting a track).

Gates: Pd tracks the Swerling-1 closed form per range bin; false-alarm
rate within 5σ. Figures: the same 15-minute PPI window as stages 6–8, and
the max-range analysis (Pd vs range + broken-track fraction vs range with
both limits marked). Note how the stochastic limits (70.5 / 39 km) sit
inside stage 8's deterministic 74.8 km horizon — fluctuation, not the
radar equation alone, is what actually breaks tracks.

All detections carry their measured `snr_db` down to the 8 dB floor, so
any CFAR threshold ≥ the floor can be applied post-hoc by filtering —
one dataset supports a full ROC sweep.

## Outputs (per stage, per day)

- `radar_truth_<date>.csv` — every beam crossing with measured SNR and
  detection outcome (the Pd denominator).
- `radar_detections_<date>.csv` — what a tracker sees: `scan_idx`, `t`,
  `range_m`, `azimuth_deg`, `snr_db`, `source` (`target`/`noise`/`clutter`),
  plus truth linkage for evaluation only.
- `measurements_summary.csv` — one row per day.
- Figures in `<data root>/plot/`: every stage emits a PPI view of the same
  15-minute window; stages 7 and 9 additionally emit a **B-scope**
  (range vs azimuth, the radar's native frame) and an **RTI** (range vs
  time over 60 min, where targets slope with range rate, clutter draws
  flat lines, and noise speckles); stage 9 adds the max-range analysis.

## Extending

All radar physics: `utils/scenario.py`. Geometry: `utils/beam_crossings.py`.
Stochastic layer and stage recipes (`MeasurementConfig`):
`utils/measurements.py`. The scenario JSON is the single source of truth —
edit it (or rerun stage 5 with flags) and rerun stages 6–9.
