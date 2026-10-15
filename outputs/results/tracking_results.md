# Deliverable 1 (tracking): YOLOv8s fine-tuned + ByteTrack, live in simulation

Person class, IoU threshold 0.5, 4 Hz perception, 120 s run, one fixed foggy night per run.

| Agent    | Frames | True boxes |   MOTA |   IDF1 |  Prec | Recall | ID sw |   FP |   FN |
|----------|--------|------------|--------|--------|-------|--------|-------|------|------|
| pole_cam |    465 |       1790 |  0.898 |  0.603 | 0.935 |  0.974 |    15 |  121 |   47 |
| ugv      |    465 |       1824 |  0.703 |  0.394 | 0.998 |  0.722 |    32 |    3 |  507 |
| uav      |    430 |        435 |  0.230 |  0.166 | 0.983 |  0.269 |    15 |    2 |  318 |
