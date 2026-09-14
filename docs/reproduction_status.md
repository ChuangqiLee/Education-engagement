# Reproduction status

Statuses distinguish code reproduction from numerical reproduction.

| Component | Paper described | Implemented | Exact numerical reproduction | Assumption / gap | Verification command |
|---|---:|---:|---|---|---|
| 2-second sampling | yes | yes | code-exact | source timing depends on codec | `pytest tests/test_video_source.py` |
| Four-source input | yes | yes | interface-exact | synchronization unspecified | `pytest tests/test_video_source.py` |
| 468 landmarks | yes | yes | impossible without original | MediaPipe/synthetic substitute | `pytest tests/test_landmarks.py` |
| Expression classifier | yes | yes | impossible without data/weights | ImageFolder baseline | `python scripts/train_expression.py --config configs/train_expression_demo.yaml` |
| Body classifier | yes | yes | impossible without data/weights | single-frame baseline | `python scripts/train_behavior.py --config configs/train_behavior_demo.yaml` |
| ResNet-34 variants | ambiguous | yes | architecture alternatives only | stem ambiguity | `pytest tests/test_models.py` |
| GoogLeNet/AlexNet/VGG19 | named | yes | impossible without protocol/data | torchvision defaults | `python scripts/evaluate_models.py --forward-only` |
| PnP pose | yes | yes | partial | camera/model geometry absent | `pytest tests/test_head_pose.py` |
| XGBoost preset/API | partial | yes | parameter-exact, results unavailable | practical feature role | `pytest tests/test_xgboost.py` |
| Fuzzy equations 6-13 | yes | yes | partial | secondary weights/S absent | `pytest tests/test_fuzzy.py` |
| G1/G2/G3 mapping | yes | yes | code-exact | head ambiguity retained | `pytest tests/test_fuzzy.py` |
| Image/video/realtime pipeline | conceptually | yes | code-only | detector/tracker/UI details absent | `pytest tests/test_pipeline.py` |
| Event logs and class report | conceptually | yes | code-only | report schema absent | `pytest tests/test_reporting.py` |
| Table 1 | yes | reference saved | impossible without data/protocol | no result fabrication | `python scripts/reproduce_table1.py` |
| Grade t-test | yes | yes | partial from summary | means are rounded | `python stats/reproduce_from_summary.py` |
| K-S/Levene | yes | raw-data path yes | impossible without raw scores | synthetic only for demo | `python stats/reproduce_from_raw.py --synthetic` |
| ANOVA consistency | yes | audited | paper values internally inconsistent | raw scores absent | `python stats/reproduce_from_summary.py` |

