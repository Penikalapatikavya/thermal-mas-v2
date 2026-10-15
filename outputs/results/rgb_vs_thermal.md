# RGB vs thermal person detection, 420 identical frames from 3 Webots missions

Night = 10% light with 10x amplified sensor noise. Fog: same attenuation for RGB and thermal (generous to RGB; visible light is scattered more than LWIR in real fog).

| Condition | RGB, COCO YOLOv8s (recall / precision) | Thermal, ours (recall / precision) |
|---|---|---|
| Day, fog 0.00 | 0.756 / 0.958 | 0.919 / 0.896 |
| Day, fog 0.03 | 0.490 / 0.972 | 0.922 / 0.954 |
| Day, fog 0.09 | 0.098 / 0.990 | 0.899 / 0.948 |
| Day, fog 0.15 | 0.072 / 1.000 | 0.306 / 0.934 |
| Night, fog 0.00 | 0.563 / 0.957 | 0.919 / 0.896 |
| Night, fog 0.03 | 0.161 / 0.918 | 0.922 / 0.954 |
| Night, fog 0.09 | 0.091 / 0.848 | 0.899 / 0.948 |
| Night, fog 0.15 | 0.061 / 0.750 | 0.306 / 0.934 |
