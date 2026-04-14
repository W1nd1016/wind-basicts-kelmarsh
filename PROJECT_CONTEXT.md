# Project Context

## Project Summary
- Project name: wind power forecasting with SCADA + CERRA multi-modal fusion.
- Core task: use past 24 hours of turbine SCADA data and CERRA meteorological data to forecast the next 6 hours of turbine power.
- Forecast granularity: targets are hourly power values for each turbine.
- Historical input granularity: SCADA and aligned CERRA analysis are sampled every 3 hours.

## Current Task Definition
- Input history length: 24 hours.
- Historical sampling interval: 3 hours.
- Number of history steps: typically `L = 9` in current code/data pipeline.
- Forecast horizon: 6 hours.
- Target sampling interval: 1 hour.
- Number of forecast steps: typically `H = 6`.

## Data Modalities

### 1. SCADA history
- Source: turbine-level SCADA observations.
- Time range: past 24 hours, sampled every 3 hours.
- Current per-step feature definition in code:
  - `P3`: normalized power
  - `dP3`: normalized power change / related power-difference feature
  - `W3`: normalized observed wind speed
  - `dir_sin`, `dir_cos`: wind direction encoded as sine/cosine
  - `nac_sin`, `nac_cos`: nacelle yaw encoded as sine/cosine
- Tensor shape in current training pipeline:
  - `x_obs`: `(B, L, N, 7)`

### 2. CERRA analysis history
- Source: historical meteorological analysis fields around each turbine.
- Time range: past 24 hours, aligned to the SCADA history grid.
- Spatial form: `K = 16` nearby grid points per turbine in the current dataset.
- Current per-grid-point feature definition:
  - speed
  - direction
  - `u`
  - `v`
- Tensor shape in current training pipeline:
  - flattened in dataset as part of `x`
  - reshaped in model to `(B, L, N, K, 4)`

### 3. CERRA forecast
- Source: forecast issued at the anchor time.
- Time range: next 6 hours.
- Spatial form: same `K = 16` nearby grid points per turbine.
- Current per-grid-point feature definition:
  - speed
  - direction
  - `u`
  - `v`
- Tensor shape in current training pipeline:
  - `fc0`: `(B, H, N, K, 4)`

## Prediction Target
- Target: future turbine power for the next 6 hourly steps.
- Tensor shape:
  - `y`: `(B, H, N)`
- Loss masking:
  - `m`: `(B, H, N)`
  - used to exclude invalid / curtailed / downtime periods from training and evaluation.

## Main Code Entry Points
- Primary training script:
  - [tools/train_s2_agcrn_FnpOnlyAnalysis.py](/home/fzh/projects/wind_basicts/tools/train_s2_agcrn_FnpOnlyAnalysis.py:1)
- Dataset:
  - [datasets/wind_dataset_scada_cerra_s2_FnpOnlyAnalysis.py](/home/fzh/projects/wind_basicts/datasets/wind_dataset_scada_cerra_s2_FnpOnlyAnalysis.py:1)
- Main fusion model:
  - [models/fnp_fusion_OnlyAnalysis_noRot_paper1_plus_v2_wake_downwind.py](/home/fzh/projects/wind_basicts/models/fnp_fusion_OnlyAnalysis_noRot_paper1_plus_v2_wake_downwind.py:1)
- S2 wrapper:
  - [models/agcrn_s2_wrapper_FnpOnlyAnalysis.py](/home/fzh/projects/wind_basicts/models/agcrn_s2_wrapper_FnpOnlyAnalysis.py:1)
- Graph recurrent decoder:
  - [models/agcrn_seq2seq_baseline2_FapOnlyAnalysis.py](/home/fzh/projects/wind_basicts/models/agcrn_seq2seq_baseline2_FapOnlyAnalysis.py:1)
- Paper/report draft:
  - [report](/home/fzh/projects/wind_basicts/report:1)

## Current Model Structure
- Overall structure: encoder-decoder.
- Encoder output:
  - history latent `z`: `(B, L, N, D)`
  - lead-wise exogenous forecast embedding `e_fc`: `(B, H, N, E)`
- Decoder output:
  - predicted power `y_hat`: `(B, H, N)`

## Current Technical Story for the Paper
- Main motivation:
  - SCADA history, CERRA analysis, and CERRA forecast are heterogeneous modalities with different temporal and spatial properties.
- Main modeling idea:
  - use FNP-style multi-modal fusion to combine these modalities instead of simple concatenation.
- Important encoder blocks:
  - `FuncRepVFR`
  - `SetConv`
  - `NeuralFourierLayer`
  - `GridEncoder`
  - `HorizonPool`
  - `TriDAMFreq`
  - `VecFiLM`
- Decoder:
  - AGCRN-style graph recurrent seq2seq decoder.

## Current Physics / Geometry Components
- SCADA wind vector construction from wind speed and direction.
- Nacelle-wind misalignment using sine/cosine form.
- Wake-directed cross-turbine mixing on SCADA latents.
- Downwind grid-point penalty for CERRA grid encoding.
- Physics alignment contexts:
  - SCADA vs analysis alignment
  - analysis vs forecast alignment

## Current Implementation Facts That Matter for Writing
- SCADA vector branch currently uses:
  - `[u, v, nac_sin, nac_cos, cos_mis, sin_mis]`
- Scalar fusion gate currently receives:
  - observation / analysis / forecast spectral magnitudes
  - plus spectral magnitudes from `ctx_obg` and `ctx_bgfc`
- Wake mixing and downwind penalty are currently enabled by default in the main fusion model.
- Angle calibration exists in code but is currently disabled by default.

## Dataset / Site Facts
- Current dataset root:
  - `data/wind_scada_cerra_v1_s2_FnpOnlyAnalysis_dataset2`
- Metadata note:
  - current `meta.json` does not expose a dataset-name field
- Current number of turbines:
  - `N = 14`
- Current number of CERRA neighboring points per turbine:
  - `K = 16`

## Working Rules For Future Sessions
- Treat this file as the stable source of project background.
- If code changes invalidate any statement here, update this file.
- When starting a new Codex session, ask Codex to read first:
  - `PROJECT_CONTEXT.md`
  - `CURRENT_STATUS.md`
- After reading `PROJECT_CONTEXT.md`, Codex should continue and read the current main training path in full:
  - the current main training script
  - the current main fusion model file
  - all directly imported local project files required by that training script
  - all directly imported local project files required by the current main fusion model
- For the current main training path, this dependency set includes at least:
  - `tools/train_s2_agcrn_FnpOnlyAnalysis.py`
  - `datasets/wind_dataset_scada_cerra_s2_FnpOnlyAnalysis.py`
  - `models/fnp_fusion_OnlyAnalysis_noRot_paper1_plus_v2_wake_downwind.py`
  - `models/fnp_fusion_OnlyAnalysis.py`
  - `models/agcrn_s2_wrapper_FnpOnlyAnalysis.py`
  - `models/agcrn_seq2seq_baseline2_FapOnlyAnalysis.py`
- The goal for a new session is:
  - after seeing this file, Codex should proactively finish reading the files above and the relevant local dependency chain before treating itself as fully initialized for this project
