# Thermal person detection vs fog (270 test images re-rendered, fine-tuned YOLOv8s, conf 0.25)

| Fog (attenuation per m) | Transmission at 25 m | Person recall | Precision |
|---|---|---|---|
| 0.00 | 100% | 0.982 | 0.902 |
| 0.03 | 47% | 0.978 | 0.958 |
| 0.06 | 22% | 0.927 | 0.954 |
| 0.09 | 11% | 0.845 | 0.960 |
| 0.15 | 2% | 0.292 | 1.000 |
| 0.25 | 0% | 0.202 | 1.000 |
| 0.40 | 0% | 0.156 | 1.000 |
| 0.60 | 0% | 0.146 | 1.000 |
