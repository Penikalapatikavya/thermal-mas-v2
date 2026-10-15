"""CLEAR MOT (MOTA, ID switches) and IDF1 from live perception logs (person class)."""
import glob
import json
import os
import numpy as np
import lap

LOG_DIR = os.path.expanduser("~/thermal-mas-v2/outputs/live_logs")
OUT = os.path.expanduser("~/thermal-mas-v2/outputs/results/tracking_results.md")
IOU_T = 0.5


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def evaluate(frames):
    n_gt = n_pred = tp = fp = fn = idsw = 0
    last_match = {}                     # gt id -> pred id it was last matched to
    pair_counts = {}                    # (gt id, pred id) -> frames overlapping (for IDF1)
    gt_frames, pred_frames = {}, {}
    for fr in frames:
        gts = [g for g in fr["gt"] if g["cls"] == 0]
        preds = [p for p in fr["pred"] if p["cls"] == "person" and p["id"] is not None]
        n_gt += len(gts)
        n_pred += len(preds)
        for g in gts:
            gt_frames[g["id"]] = gt_frames.get(g["id"], 0) + 1
        for p in preds:
            pred_frames[p["id"]] = pred_frames.get(p["id"], 0) + 1
        ious = np.array([[iou(g["xyxy"], p["xyxy"]) for p in preds] for g in gts]) \
            if gts and preds else np.zeros((len(gts), len(preds)))
        for gi, g in enumerate(gts):
            for pi, p in enumerate(preds):
                if ious[gi, pi] >= IOU_T:
                    k = (g["id"], p["id"])
                    pair_counts[k] = pair_counts.get(k, 0) + 1
        matches = {}
        # keep previous associations if still valid (CLEAR MOT rule)
        for gi, g in enumerate(gts):
            prev = last_match.get(g["id"])
            for pi, p in enumerate(preds):
                if p["id"] == prev and ious[gi, pi] >= IOU_T and pi not in matches.values():
                    matches[gi] = pi
        free_g = [gi for gi in range(len(gts)) if gi not in matches]
        free_p = [pi for pi in range(len(preds)) if pi not in matches.values()]
        if free_g and free_p:
            cost = np.array([[1 - ious[gi, pi] for pi in free_p] for gi in free_g])
            _, x, _ = lap.lapjv(cost, extend_cost=True, cost_limit=1 - IOU_T)
            for a, b in enumerate(x):
                if b >= 0:
                    matches[free_g[a]] = free_p[b]
        for gi, pi in matches.items():
            gid, pid = gts[gi]["id"], preds[pi]["id"]
            if gid in last_match and last_match[gid] != pid:
                idsw += 1
            last_match[gid] = pid
        tp += len(matches)
        fn += len(gts) - len(matches)
        fp += len(preds) - len(matches)
    # IDF1: best one-to-one assignment of ground-truth IDs to predicted IDs
    gids, pids = sorted(gt_frames), sorted(pred_frames)
    idtp = 0
    if gids and pids:
        w = np.zeros((len(gids), len(pids)))
        for (g, p), c in pair_counts.items():
            w[gids.index(g), pids.index(p)] = c
        _, x, _ = lap.lapjv(w.max() - w, extend_cost=True)
        idtp = sum(w[i, j] for i, j in enumerate(x) if j >= 0)
    mota = 1 - (fn + fp + idsw) / n_gt if n_gt else float("nan")
    idf1 = 2 * idtp / (n_gt + n_pred) if (n_gt + n_pred) else float("nan")
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    return dict(frames=len(frames), gt=n_gt, mota=mota, idf1=idf1, precision=prec,
                recall=rec, idsw=idsw, fp=fp, fn=fn, gt_ids=len(gids), pred_ids=len(pids))


rows, all_frames = [], {}
for agent in ("pole_cam", "ugv", "uav"):
    logs = sorted(glob.glob(os.path.join(LOG_DIR, f"{agent}_*.jsonl")))
    if not logs:
        continue
    path = logs[-1]
    frames = [json.loads(l) for l in open(path) if l.strip()]
    rows.append((agent, os.path.basename(path), evaluate(frames)))

hdr = f"| {'Agent':<8} | {'Frames':>6} | {'True boxes':>10} | {'MOTA':>6} | {'IDF1':>6} | {'Prec':>5} | {'Recall':>6} | {'ID sw':>5} | {'FP':>4} | {'FN':>4} |"
sep = "|" + "|".join("-" * (len(c)) for c in hdr.split("|")[1:-1]) + "|"
lines = [hdr, sep]
for agent, _, r in rows:
    lines.append(f"| {agent:<8} | {r['frames']:>6} | {r['gt']:>10} | {r['mota']:>6.3f} | {r['idf1']:>6.3f} | "
                 f"{r['precision']:>5.3f} | {r['recall']:>6.3f} | {r['idsw']:>5} | {r['fp']:>4} | {r['fn']:>4} |")
table = "\n".join(lines)
print(table)
for agent, name, r in rows:
    print(f"{agent}: log {name}, true person IDs {r['gt_ids']}, predicted track IDs {r['pred_ids']}")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    f.write("# Deliverable 1 (tracking): YOLOv8s fine-tuned + ByteTrack, live in simulation\n\n")
    f.write("Person class, IoU threshold 0.5, 4 Hz perception, 120 s run, one fixed foggy night per run.\n\n")
    f.write(table + "\n")
print("saved", OUT)
