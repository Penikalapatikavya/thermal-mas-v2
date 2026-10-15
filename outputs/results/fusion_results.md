# Deliverable 3: belief fusion on the world map (offline replay of one live run)

Error model (bias included): {"pole_cam": {"range_rel_sigma": 0.0346, "lateral_sigma_m": 0.133}, "ugv": {"range_rel_sigma": 0.162, "lateral_sigma_m": 0.14}, "uav": {"range_rel_sigma": 0.0354, "lateral_sigma_m": 0.1}}

| Agents | Coverage of people | Map error median (m) | Map error mean (m) | False tracks per moment | of which duplicates (<3 m) | ID switches | Track IDs (4 people) |
|---|---|---|---|---|---|---|---|
| pole_cam | 0.996 | 0.72 | 0.71 | 0.00 | 0.00 | 0 | 4 |
| ugv | 0.000 | nan | nan | 0.00 | 0.00 | 0 | 0 |
| uav | 0.348 | 0.30 | 0.43 | 0.18 | 0.49 | 12 | 13 |
| pole_cam + ugv | 1.000 | 0.08 | 0.19 | 0.04 | 0.50 | 1 | 5 |
| pole_cam + uav | 0.996 | 0.60 | 0.58 | 0.02 | 1.00 | 2 | 5 |
| ugv + uav | 0.464 | 0.27 | 0.47 | 0.03 | 1.00 | 5 | 8 |
| pole_cam + ugv + uav | 1.000 | 0.08 | 0.16 | 0.01 | 1.00 | 0 | 4 |

## Communication loss (5 random seeds per loss level, averaged)

| Receiving agent | Message loss | Coverage | Map error median (m) | Map error mean (m) | False tracks | ID switches |
|---|---|---|---|---|---|---|
| pole_cam | 0% | 1.000 | 0.08 | 0.16 | 0.01 | 0.0 |
| pole_cam | 20% | 0.999 | 0.10 | 0.17 | 0.01 | 0.8 |
| pole_cam | 50% | 0.998 | 0.12 | 0.22 | 0.01 | 1.2 |
| pole_cam | 80% | 0.997 | 0.23 | 0.37 | 0.00 | 0.4 |
| pole_cam | 100% | 0.996 | 0.72 | 0.71 | 0.00 | 0.0 |
| ugv | 0% | 1.000 | 0.08 | 0.16 | 0.01 | 0.0 |
| ugv | 20% | 0.999 | 0.08 | 0.16 | 0.01 | 0.0 |
| ugv | 50% | 0.994 | 0.08 | 0.17 | 0.01 | 0.0 |
| ugv | 80% | 0.949 | 0.11 | 0.23 | 0.10 | 2.0 |
| ugv | 100% | 0.000 | nan | nan | 0.00 | 0.0 |
| uav | 0% | 1.000 | 0.08 | 0.16 | 0.01 | 0.0 |
| uav | 20% | 0.999 | 0.09 | 0.17 | 0.01 | 0.0 |
| uav | 50% | 0.994 | 0.11 | 0.21 | 0.02 | 0.8 |
| uav | 80% | 0.942 | 0.18 | 0.33 | 0.09 | 1.2 |
| uav | 100% | 0.348 | 0.30 | 0.43 | 0.18 | 12.0 |
