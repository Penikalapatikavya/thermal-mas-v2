# Multi-Agent Depth-Aware Autonomous Decision System Using Infrared Vision

B.Tech Final Year Project (23AID498), Amrita Vishwa Vidyapeetham, Amritapuri.
Team: Kavya Penikalapati, Akshaya Giri Guduru, Nagadeep Reddy. Guide: Dr. Divya Udayan J.

Night-time search and rescue in Webots: a pole camera, a ground robot (Pioneer 3-DX) and a drone (Mavic 2 Pro)
use simulated thermal cameras to detect (YOLOv8s), track (BoT-SORT) and localise (depth U-Net + ground plane)
people, share observations over radio, fuse them with Kalman filters, and decide with rule-based or MAPPO policies.

## Structure
- `worlds/` Webots world · `controllers/` robot programs · `src/` shared code
- `training/` dataset, evaluation, batch runs, RL training, experiments
- `weights/` trained models · `outputs/results/` all results · `dashboard/` Streamlit dashboard

## Run
- Simulation: open `worlds/rescue_city.wbt` in Webots (`POLICY=rl RL_TAG=_dr` for the MAPPO team)
- Batch evaluation: `python3 training/run_batch.py <label> <first_seed> <runs>`
- Results: `python3 training/summarize_batch.py rule_v5 rl_v2`
- Dashboard: `streamlit run dashboard/app.py`

Datasets and raw logs are not in this repository (size); they are archived separately.
