# Paper summary for reproduction

Source: Li, Weng, Li, and Zhang, *Multimodal Learning Engagement Assessment
System: An Innovative Approach to Optimizing Learning Engagement*, IJHCI 41(5),
3474-3490, DOI 10.1080/10447318.2024.2338616.

The proposed system has four named stages: image acquisition, keypoint
detection, multimodal recognition, and engagement analysis. Four HD classroom
cameras provide a frame every two seconds. The article describes XGBoost as
detecting 468 facial/body feature points, ResNet-34 as classifying expression
and body behavior, OpenCV PnP as estimating head rotation, and fuzzy
comprehensive evaluation as combining expression, head pose, and behavior.

The three primary fuzzy weights are 0.4, 0.3, and 0.3. Outputs are active (G1),
partial (G2), and passive (G3). The implementation study compares two classes
containing 234 students in total using exam grades.

This repository reproduces the disclosed software architecture and exposes
undisclosed choices as configuration. It does not claim to reconstruct the
authors' trained weights or raw-data results.

