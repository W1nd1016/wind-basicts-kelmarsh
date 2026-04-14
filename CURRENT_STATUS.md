# Current Status

## Purpose
- This file stores the latest project state that may change frequently.
- Update this file whenever the main training path, model logic, or paper wording changes in a meaningful way.

## Current Main Entry
- Main training script:
  - `tools/train_s2_agcrn_FnpOnlyAnalysis.py`
- Main fusion model:
  - `models/fnp_fusion_OnlyAnalysis_noRot_paper1_plus_v2_wake_downwind.py`
- Main report file:
  - `report`

## Current Active Goal
- Example:
  - finalize the method section so it matches the current implementation
  - run/compare ablations
  - prepare paper figures and experiment tables

## Latest Important Code State
- Date:
  - 2026-04-13
- Current facts:
  - SCADA vector branch includes `u, v, nac_sin, nac_cos, cos_mis, sin_mis`
  - scalar fusion gate consumes `ctx_obg` and `ctx_bgfc`
  - wake mixing default is enabled
  - downwind penalty default is enabled
  - angle calibration exists but default is disabled

## Report Sync Notes
- Statements that should currently be true in the paper:
  - SCADA direction-related branch includes wind vector and nacelle/misalignment cues
  - scalar frequency-domain fusion is conditioned on physics contexts
  - wake-directed mixing and downwind penalty are part of the current model
- Statements that still require care:
  - SCADA missing values are not handled by an explicit SCADA valid-mask path inside `FuncRepVFR`
  - report currently contains duplicated method sections and repeated LaTeX labels that should be cleaned later

## Open Questions / Risks
- Example items to keep updated:
  - whether the report text exactly matches the newest code
  - whether all backup files are obsolete or still referenced
  - whether normalization statements in the paper fully match preprocessing code

## Recent Changes Log
- 2026-04-13
  - Enabled and integrated physics contexts into scalar tri-branch fusion call.
  - Expanded SCADA vector branch to 6-D input with nacelle and misalignment cues.
  - Confirmed wake mixing and downwind penalty are on by default in the main fusion model.

## How To Start A New Codex Session
- Recommended opening prompt:
  - `Please first read PROJECT_CONTEXT.md, CURRENT_STATUS.md, tools/train_s2_agcrn_FnpOnlyAnalysis.py, and models/fnp_fusion_OnlyAnalysis_noRot_paper1_plus_v2_wake_downwind.py. Use them as the current source of truth for this project before doing anything else.`

## What To Update After Major Changes
- Update this file if you change:
  - the main training script
  - the main model structure
  - tensor definitions used in the paper
  - default feature toggles
  - the active writing goal
