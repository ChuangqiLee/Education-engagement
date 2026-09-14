# Engineering assumptions

1. MediaPipe Face Mesh is the practical 468-point backend; it is not claimed as
   the authors' implementation.
2. A synthetic 468-point detector is used only for deterministic smoke tests.
3. A configurable generic six-point 3D face and approximate camera matrix are
   used until real calibration is supplied.
4. Torsion is operationalized as yaw for the default threshold rule.
5. Expression/body baselines are independent single-frame classifiers.
6. Centroid association is a replaceable baseline tracker; it is not suitable
   for identity-critical longitudinal research.
7. Demo fuzzy secondary weights are equal and S=[100,60,20].
8. Demo predictions and synthetic images/videos are prominently marked
   `DEMO / UNTRAINED` and never reported as paper accuracy.
9. Torchvision defaults provide all architectures except the explicitly
   modified paper-compatible ResNet stem.
10. The default seed is 42 solely for repeatability; the paper gives no seed.

MediaPipe 0.10.21 was selected for the available Python 3.9/macOS environment.
On macOS it may create an OpenGL context even for a CPU graph; headless/sandboxed
deployments should use a supported Tasks backend or grant the required graphics
context. This environment passed an operational import/process smoke test.
