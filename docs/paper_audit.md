# Paper audit

This is an evidence/provenance ledger. Labels mean:

- **[PAPER]** explicitly stated or tabulated in the article.
- **[INFERENCE]** a bounded interpretation of disclosed material.
- **[ASSUMPTION]** an engineering choice needed to execute code.
- **[UNRESOLVED]** cannot be recovered from the article alone.

## System and data acquisition

- [PAPER] Four stages: image acquisition, keypoint detection, multimodal
  recognition, engagement analysis (Figure 1).
- [PAPER] Four HD cameras: three at the classroom front (middle, left, right)
  and one above the classroom center (Section 2.1).
- [PAPER] Each camera yields one sampled frame every two seconds (Section 2.2).
- [UNRESOLVED] Camera make, resolution, frame rate, lens, calibration,
  synchronization method, student-to-camera association, occlusion handling,
  retention policy, and video/annotation release.
- [ASSUMPTION] Practical inputs support image, file video, USB index, and RTSP;
  timestamps use source frame time where possible.

## Landmarks and XGBoost

- [PAPER] The abstract says 468 facial and bodily features are captured using
  XGBoost. Sections 2 and 4 say XGBoost detects 468 key facial feature points.
- [PAPER] XGBoost preset: learning rate 0.1, 100 estimators, max depth 5,
  min_child_weight 3, subsample 0.9, colsample_bytree 0.8.
- [UNRESOLVED] No training samples, target labels, feature representation,
  objective, landmark topology, detector architecture, coordinate convention,
  or method showing how regression trees localize 468 points is supplied.
- [UNRESOLVED] The wording alternates between facial points and facial/body
  features, while Figure 4 visually resembles a dense face mesh.
- [ASSUMPTION] `mediapipe_facemesh` operationalizes the 468-landmark claim in
  practical mode. This is not explicitly specified by the paper.
- [ASSUMPTION] Practical XGBoost consumes normalized geometry statistics
  derived from an already detected mesh. Paper-faithful mode fails explicitly
  where the missing original detector would otherwise be required.

## Facial expression and body behavior

- [PAPER] Section 3 first lists eight general expression categories, then uses
  learning-specific happiness, confusion, and boredom.
- [PAPER] Table 4 instead uses Happy, Serious, and Bored, mapped to positive/G1,
  neutral/G2, and negative/G3.
- [INFERENCE] Canonical labels are `positive`, `neutral`, `negative`; aliases
  map `happy -> positive`, `serious/confusion/confused -> neutral`, and
  `bored/boredom -> negative` without erasing the inconsistency.
- [PAPER] Table 6 body categories and grades: nodding G1, writing G1,
  turn head G2, phone G3, sleeping G3.
- [PAPER] Table 2 additionally says "Lectures", "Turned", "Look at your
  phone", "Sleep", and "Runaway"; these do not form the same taxonomy as
  Table 6.
- [UNRESOLVED] Image crop policy, input resolution, augmentations, annotations,
  temporal windowing, multi-label policy, and data split.
- [ASSUMPTION] The implemented baseline treats both tasks as single-image,
  single-label ImageFolder classification and reserves a temporal interface.

## ResNet and model comparison

- [PAPER] A 34-layer ResNet is the selected model. The prose specifies a 3x3
  convolution, stride 1, and same padding.
- [PAPER] Figure 6 resembles a standard ResNet-34 pipeline that ordinarily has
  a 7x7/stride-2 stem and max pool.
- [UNRESOLVED] It is unclear whether the 3x3 statement applies to the stem, all
  residual convolutions, or a modified architecture.
- [ASSUMPTION] Both `resnet34_standard` and
  `resnet34_paper_compatible` (3x3/stride-1 stem, no max pool) are exposed.
- [PAPER] Compared families: ResNet, GoogLeNet, AlexNet, VGG. Table 1 values:
  GoogLeNet 85.56; GoogLeNet batch 64 84.81; ResNet-34 97.59;
  `resnet34_test_batch_size80` 70.44 at step 5; another ResNet-34 batch 80
  run 98.00; VGG-19 81.77.
- [UNRESOLVED] Dataset, exact split, number of classes, label counts,
  preprocessing, optimizer, model initialization, learning-rate schedule,
  validation protocol, definition of "Value", and full seeds/checkpoints.
- [PAPER] Table 1 mixes final/model names, batch variants, different steps and
  wall times, so it is not a controlled comparison table by itself.

## Head pose

- [PAPER] OpenCV PnP uses facial 3D feature points and camera parameters,
  returns a rotation vector, then Euler pitch/yaw/roll.
- [PAPER] Reported observed ranges: pitch -60.4 to 69.6, yaw -79.8 to 75.3,
  roll -40.9 to 63.3 degrees.
- [PAPER] Table 2 defines torsion `a` and pitch `b`; cutoffs are |a|=23.6 and
  |b|=25.4. Exceeding either tends toward G3; staying below both gives G1.
- [UNRESOLVED] "torsion" is not tied unambiguously to yaw or roll. Table 5 uses
  threshold labels in a visually ambiguous form and only produces G1/G3.
- [ASSUMPTION] Configuration calls 23.6 the yaw-or-torsion threshold. Generic
  six-point 3D geometry and focal-length=image-width intrinsics are documented
  defaults, replaceable by calibrated values.

## Fuzzy comprehensive evaluation

- [PAPER] Primary factors F1/F2/F3 are expression/head pose/body behavior.
- [PAPER] Grades are G1 active, G2 partial, G3 passive; corresponding numerical
  assignments are named S1/S2/S3 (Tables 3-7).
- [PAPER] R11 is 3x3, R12 is 4x3, R13 is 5x3 (Equations 6-8).
- [PAPER] B11=W1R11, B12=W2R12, B13=W3R13 (Equations 9-11).
- [PAPER] D=[0.4,0.3,0.3]B (Equation 12) and Y=D[S1,S2,S3]^T
  (Equation 13). The primary weights may be changed by context.
- [UNRESOLVED] No numerical secondary weights W1/W2/W3, membership matrices,
  membership elicitation method, or actual S1/S2/S3 percentage scores appear.
- [UNRESOLVED] Section 10 says engagement was determined with Formula (14),
  but no Formula 14 appears in the article.
- [ASSUMPTION] Demo values use equal secondary weights, crisp/probabilistic
  memberships, and score vector [100,60,20]. The paper preset leaves unknowns
  null and cannot evaluate until supplied.

## Statistical experiment

- [PAPER] 234 undergraduates, aged 18-22, in Probability and Mathematical
  Statistics; classes act as experimental/control groups. Which named class
  received the system is inferable from the prose/means but not stated in a
  clean allocation table.
- [PAPER] Class 1: n=116, mean 83.84, SD 9.29336. Class 2: n=118, mean 81.21,
  SD 9.50577. K-S p=.076 and .200 (the latter a lower bound).
- [PAPER] Equal-variance t-test: Levene F=.002 p=.967; t=2.134, df=232,
  two-sided p=.034, mean difference=2.62324, SE=1.22918, 95% CI
  [.20146, 5.04502]. Unequal-variance row: t=2.135, df=231.993, p=.034.
- [PAPER] One-way ANOVA: between SS=380.435, within SS=20574.176, total
  SS=20954.611, df 1/232/233, MS 380.435/88.682, F=4.290, p=.039.
- [UNRESOLVED] Raw grades are absent, so K-S, Levene, histograms, and the exact
  raw-data tests cannot be independently reproduced.
- [PAPER inconsistency] Rounded group means differ by 2.63, not 2.62324.
- [PAPER inconsistency] For exactly two groups, an ordinary equal-variance
  one-way ANOVA should satisfy F=t^2. Reported t^2 is about 4.554, not 4.290.
  The ANOVA between-group SS also implies a different mean gap from the t table.
- [INFERENCE] Summary-statistic t and Welch tests can be recomputed; raw-data
  normality and Levene tests cannot.

## Operational output, hardware, and ethics

- [PAPER] NVIDIA GeForce RTX 3060 described as 16 GB VRAM, R7-5800H CPU,
  Python, PyTorch.
- [PAPER] A 20 cm x 35 cm podium display shows real-time recognition results;
  after class the system generates a comprehensive teacher report with
  personalized recommendations.
- [UNRESOLVED] Software/package versions, GPU variant (consumer RTX 3060
  variants are not normally described by this exact laptop-like pairing),
  report schema/UI source, latency, privacy mechanics, consent artifacts,
  deployment topology, and source code/weights.
- [ASSUMPTION] This reproduction provides OpenCV overlay plus CSV, JSON, HTML,
  and plots. It does not require the physical display.

