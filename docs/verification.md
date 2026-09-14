# Verification record

Date: 2026-09-14

Environment used for the checked run:

- Python 3.9.6 (the project recommends 3.10/3.11 but supports 3.9)
- NumPy 1.26.4, SciPy 1.13.1, scikit-learn 1.5.2
- OpenCV 4.11.0, PyTorch 2.6.0, torchvision 0.21.0
- XGBoost 2.1.4, pandas 2.2.3, Matplotlib 3.9.4
- MediaPipe 0.10.21 with protobuf 4.25.9

## Completed checks

| Check | Result |
|---|---|
| package wheel build and isolated-prefix install | passed |
| Python compileall | passed |
| import smoke test | passed |
| full pytest suite | 19 passed |
| synthetic ImageFolder generation | passed; explicit marker written |
| duplicate/corrupt/missing image validation | passed; zero duplicates |
| tiny expression training | passed; synthetic checkpoint/result written |
| tiny body behavior training | passed; synthetic checkpoint/result written |
| ResNet variants, GoogLeNet, AlexNet, VGG19 forward | all returned `[1,3]` |
| OpenCV PnP tests | frontal and rotated projections passed |
| XGBoost fit/predict/proba/save/load | passed |
| MediaPipe FaceMesh initialize/process | passed; procedural non-face returned zero detections |
| fuzzy matrices, rules, score, unresolved paper score | passed |
| one-image pipeline | passed; one visibly marked demo record |
| sampled video pipeline | passed; three records plus annotated MP4/report |
| source/realtime loop | passed on generated video |
| summary-statistic reproduction | passed |
| synthetic raw-statistics pipeline | passed; output marked synthetic |
| fuzzy primary-weight sensitivity sweep | passed; 66 scenarios |

The test suite emitted 14 Matplotlib/pyparsing deprecation warnings; no test or
runtime check failed. These warnings do not alter computed results.

## Important result provenance

The tiny vision checkpoints, XGBoost training metrics, raw-statistics results,
image/video annotations, and fuzzy sensitivity table use synthetic/demo inputs.
They validate execution only. They are not reproduced paper performance.

`paper_reference_results.csv` is a transcription of Table 1. Local result files
are separate. Summary-statistic checks recomputed t=2.139637 and p=0.033428 from
the rounded published n/mean/SD values. They also found t-squared=4.578045 and
between-group SS=404.609091, inconsistent with the reported ANOVA F=4.290 and
between-group SS=380.435. No synthetic data was adjusted to conceal this gap.

