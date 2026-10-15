# Deliverable 1: IR object detection (YOLOv8s), fine-tuning results

Start model: YOLOv8s trained on FLIR ADAS v2 (best_kaggle.pt)
Fine-tuning: 30 epochs, SGD lr0=0.002, cosine LR, imgsz 640, batch 16, Colab T4 (41 min)
Training data: 1,862 simulation images + 3,000 random FLIR training images
Simulation split: 30-frame blocks per agent (leakage-free), test set never used in training

## Simulation test set (270 images: 90 pole camera, 90 UGV, 90 UAV)
| Class  | mAP50 before | mAP50 after | mAP50-95 before | mAP50-95 after |
|--------|--------------|-------------|-----------------|----------------|
| All    | 0.592        | 0.967       | 0.417           | 0.833          |
| Person | 0.814        | 0.981       | 0.598           | 0.873          |
| Car    | 0.371        | 0.953       | 0.236           | 0.792          |

## Real FLIR validation set (1,096 images): forgetting check
| Class  | mAP50 before | mAP50 after | mAP50-95 before | mAP50-95 after |
|--------|--------------|-------------|-----------------|----------------|
| All    | 0.740        | 0.740       | 0.483           | 0.480          |
| Person | 0.818        | 0.812       | 0.483           | 0.477          |
| Car    | 0.864        | 0.861       | 0.621           | 0.617          |
| Bike   | 0.538        | 0.547       | 0.345           | 0.346          |

## Thermal model ablation (FLIR model on simulation, no fine-tuning)
| Thermal model version                     | Person mAP50 | Car mAP50 |
|-------------------------------------------|--------------|-----------|
| Old project (flat colours)                | 0 detections | -         |
| Physics model, uniform car temperature    | 0.819        | 0.0004    |
| Physics model, car heat pattern           | 0.814        | 0.371     |

## Caveat
All simulation data comes from one intersection; simulation scores measure
in-scene generalisation across time, layouts, agents and thermal conditions,
not generalisation to unseen locations.
