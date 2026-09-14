# Data required for exact numerical reproduction

## Vision

- Original synchronized feeds from all four cameras, calibration and timing.
- Student/video identifiers, consent-compatible de-identification mapping, and
  class/session metadata.
- Face/body boxes or the procedure that generated them.
- All 468-point targets/topology and XGBoost training targets.
- Expression and behavior labels with annotator instructions and agreement.
- Exact train/validation/test partition at student or video level.
- Preprocessing, augmentations, optimizer/scheduler, epochs, seed, stopping
  rule, and all trained checkpoints.

## Engagement evaluation

- Numerical W1/W2/W3 secondary weights.
- R11/R12/R13 membership values or the procedure to estimate them.
- S1/S2/S3 numerical scores and the missing Formula 14.
- Student tracking and multi-camera identity-fusion method.

## Statistics

- A row-level CSV with at least `student_id, class, system_condition, score`.
- Exclusions/missing-data policy and the precise SPSS test settings, including
  the normality-test correction (the article's footnote says
  "Raleigh-corrected", likely a wording/typesetting issue).

