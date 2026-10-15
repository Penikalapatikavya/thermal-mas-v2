"""Replay saved live frames offline: detector-only vs different trackers."""
import glob
import json
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import lap
from PIL import Image
from ultralytics import YOLO
from mot_metrics import evaluate, iou

WEIGHTS = os.path.expanduser("~/thermal-mas-v2/weights/yolo_thermal_finetuned.pt")
LOG_DIR = os.path.expanduser("~/thermal-mas-v2/outputs/live_logs")
OUT = os.path.expanduser("~/thermal-mas-v2/outputs/results/tracking_replay.md")
SETUPS = [("ByteTrack conf0.25", "bytetrack.yaml", 0.25),
          ("ByteTrack conf0.10", "bytetrack.yaml", 0.10),
          ("BoT-SORT  conf0.10", "botsort.yaml", 0.10)]


def latest_run(agent):
    for path in sorted(glob.glob(os.path.join(LOG_DIR, f"{agent}_*.jsonl")), reverse=True):
        if os.path.isdir(path[:-6] + "_frames"):
            meta = [json.loads(l) for l in open(path) if l.strip()]
            return meta, path[:-6] + "_frames"
    return None, None


def frames_iter(meta, frame_dir):
    for m in meta:
        img = np.array(Image.open(os.path.join(frame_dir, m["frame"])))
        yield m, np.repeat(img[:, :, None], 3, axis=2)


def detector_only(meta, frame_dir, conf=0.25):
    model = YOLO(WEIGHTS)
    tp = fp = fn = 0
    for m, img in frames_iter(meta, frame_dir):
        r = model.predict(img, conf=conf, device="mps", verbose=False)[0]
        preds = [b for b, c in zip(r.boxes.xyxy.tolist(), r.boxes.cls.int().tolist()) if model.names[c] == "person"]
        gts = [g["xyxy"] for g in m["gt"] if g["cls"] == 0]
        matched = 0
        if gts and preds:
            cost = np.array([[1 - iou(g, p) for p in preds] for g in gts])
            _, x, _ = lap.lapjv(cost, extend_cost=True, cost_limit=0.5)
            matched = int(sum(1 for j in x if j >= 0))
        tp += matched
        fn += len(gts) - matched
        fp += len(preds) - matched
    return tp / max(1, tp + fp), tp / max(1, tp + fn)


def tracked(meta, frame_dir, tracker, conf):
    model = YOLO(WEIGHTS)
    out = []
    for m, img in frames_iter(meta, frame_dir):
        r = model.track(img, persist=True, tracker=tracker, conf=conf, device="mps", verbose=False)[0]
        preds = []
        b = r.boxes
        if b is not None and len(b) and b.id is not None:
            for tid, c, cf, xyxy in zip(b.id.int().tolist(), b.cls.int().tolist(), b.conf.tolist(), b.xyxy.tolist()):
                preds.append({"id": tid, "cls": model.names[c], "conf": cf, "xyxy": xyxy})
        out.append({"gt": m["gt"], "pred": preds})
    return evaluate(out)


lines = ["| Agent | Setup | MOTA | IDF1 | Precision | Recall | ID switches | Track IDs |",
         "|-------|-------|------|------|-----------|--------|-------------|-----------|"]
for agent in ("pole_cam", "ugv", "uav"):
    meta, frame_dir = latest_run(agent)
    if meta is None:
        print(agent, "no saved frames found")
        continue
    p, r = detector_only(meta, frame_dir)
    row = f"| {agent} | Detector only conf0.25 | - | - | {p:.3f} | {r:.3f} | - | - |"
    print(row)
    lines.append(row)
    for name, trk, conf in SETUPS:
        m = tracked(meta, frame_dir, trk, conf)
        row = (f"| {agent} | {name} | {m['mota']:.3f} | {m['idf1']:.3f} | {m['precision']:.3f} | "
               f"{m['recall']:.3f} | {m['idsw']} | {m['pred_ids']} (true {m['gt_ids']}) |")
        print(row)
        lines.append(row)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    f.write("# Tracking replay: same live frames, different setups (person class, IoU 0.5)\n\n")
    f.write("\n".join(lines) + "\n")
print("saved", OUT)
