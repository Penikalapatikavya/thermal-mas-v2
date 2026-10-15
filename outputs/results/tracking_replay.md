# Tracking replay: same live frames, different setups (person class, IoU 0.5)

| Agent | Setup | MOTA | IDF1 | Precision | Recall | ID switches | Track IDs |
|-------|-------|------|------|-----------|--------|-------------|-----------|
| pole_cam | Detector only conf0.25 | - | - | 0.961 | 1.000 | - | - |
| pole_cam | ByteTrack conf0.25 | 0.959 | 0.727 | 0.962 | 0.999 | 2 | 6 (true 4) |
| pole_cam | ByteTrack conf0.10 | 0.959 | 0.727 | 0.962 | 0.999 | 2 | 6 (true 4) |
| pole_cam | BoT-SORT  conf0.10 | 0.959 | 0.727 | 0.962 | 0.999 | 2 | 6 (true 4) |
| ugv | Detector only conf0.25 | - | - | 0.988 | 0.978 | - | - |
| ugv | ByteTrack conf0.25 | 0.816 | 0.582 | 0.997 | 0.842 | 32 | 21 (true 4) |
| ugv | ByteTrack conf0.10 | 0.817 | 0.582 | 0.997 | 0.843 | 32 | 21 (true 4) |
| ugv | BoT-SORT  conf0.10 | 0.881 | 0.551 | 0.999 | 0.907 | 34 | 29 (true 4) |
| uav | Detector only conf0.25 | - | - | 0.974 | 0.979 | - | - |
| uav | ByteTrack conf0.25 | 0.227 | 0.186 | 0.976 | 0.262 | 13 | 15 (true 4) |
| uav | ByteTrack conf0.10 | 0.227 | 0.186 | 0.976 | 0.262 | 13 | 15 (true 4) |
| uav | BoT-SORT  conf0.10 | 0.633 | 0.193 | 1.000 | 0.715 | 38 | 41 (true 4) |

## Findings
- Detector alone: precision 0.96-0.99, recall 0.98-1.00 on all three agents (foggy night, fog 0.032).
  Tracking losses therefore come from association, not detection.
- Moving cameras break motion prediction in ByteTrack (static-camera assumption):
  UAV recall 0.262 -> 0.715 and MOTA 0.227 -> 0.633 with BoT-SORT camera-motion compensation (GMC).
- Using low-confidence detections (conf 0.10 vs 0.25) had no effect in this run.
- Static pole camera: MOTA 0.959; its 2 ID switches are scenario teleports.
- Remaining issue: ID fragmentation on moving agents (UAV 41 IDs for 4 people). GMC often fails on
  low-texture thermal frames ("not enough matching points"). Addressed downstream by tracking in
  world coordinates using agent pose + depth (belief fusion stage).
- Selected for the live system: YOLOv8s (fine-tuned) + BoT-SORT.
