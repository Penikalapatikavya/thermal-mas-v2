# Deliverable 2: monocular thermal depth (U-Net, ResNet-34 encoder, supervised on simulation)

Training: 1,862 sim images (320x256), scale-invariant log loss, AdamW 3e-4 one-cycle, 40 epochs, Colab T4 (14 min)
Evaluation: 270 test images, sky / beyond-range pixels (>= 99.5 m) excluded

| Agent    | Method              | RMSE (m) | AbsRel | d1    |
|----------|---------------------|----------|--------|-------|
| all      | U-Net (ours)        | 1.89     | 0.028  | 0.988 |
| all      | mean-depth baseline | 4.15     | 0.086  | 0.886 |
| pole_cam | U-Net (ours)        | 1.42     | 0.019  | 0.991 |
| pole_cam | mean-depth baseline | 0.22     | 0.001  | 1.000 |
| ugv      | U-Net (ours)        | 3.26     | 0.032  | 0.980 |
| ugv      | mean-depth baseline | 9.07     | 0.110  | 0.868 |
| uav      | U-Net (ours)        | 1.00     | 0.034  | 0.992 |
| uav      | mean-depth baseline | 3.17     | 0.147  | 0.790 |

## Distance to each person (768 people, truth = depth at the person's hottest pixels)
| Estimator           | Mean abs err (m) | Median abs err (m) | Mean rel. err | Within 10% |
|---------------------|------------------|--------------------|---------------|------------|
| box median          | 3.59             | 1.54               | 0.156         | 0.643      |
| box 20th percentile | 1.56             | 0.86               | 0.064         | 0.785      |
| hottest pixels      | 1.80             | 1.21               | 0.075         | 0.764      |
Selected: box 20th percentile.

## Notes
- Static pole camera: background depth is constant, so a calibrated background map beats the network
  (0.22 m vs 1.42 m). Real fixed cameras are calibrated at installation; mobile agents use the network.
- Supervised on simulator depth instead of self-supervised (Shin et al.), since exact depth labels are available.
- Runs at ~25 ms/frame on a MacBook Air M2 (MPS).
