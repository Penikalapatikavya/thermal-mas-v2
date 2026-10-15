"""Thermal multi-agent rescue: results dashboard.   Run: streamlit run dashboard/app.py"""
import glob, json, os, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

ROOT = os.path.expanduser("~/thermal-mas-v2")
RES = f"{ROOT}/outputs/results"
st.set_page_config(page_title="Thermal multi-agent rescue", layout="wide")


# ---------------- helpers ----------------
def show_md(name, title=None):
    p = f"{RES}/{name}"
    if title:
        st.subheader(title)
    if os.path.exists(p):
        st.markdown(open(p).read())
    else:
        st.info(f"{name} not found in outputs/results")


def md_table(name):
    p = f"{RES}/{name}"
    if not os.path.exists(p):
        return None
    rows = [l.strip() for l in open(p) if l.strip().startswith("|")]
    if len(rows) < 3:
        return None
    head = [c.strip() for c in rows[0].strip("|").split("|")]
    body = [[c.strip() for c in r.strip("|").split("|")] for r in rows[2:]]
    return pd.DataFrame(body, columns=head)


def batches():
    return sorted(os.path.basename(p)[6:-4] for p in glob.glob(f"{RES}/batch_*.csv"))


@st.cache_data
def load_batch(label):
    p = f"{RES}/batch_{label}.csv"
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p)
    if "sim_fault" in df:
        df = df[df["sim_fault"].astype(str) != "True"]
    return df


def ssum(df, col):
    return int(np.nansum(pd.to_numeric(df[col], errors="coerce"))) if col in df else 0


def summary(labels):
    rows = []
    for l in labels:
        df = load_batch(l)
        if df is None or df.empty:
            continue
        ok = df["success"].astype(str) == "True"
        t = pd.to_numeric(df.get("time_all"), errors="coerce")
        rows.append({"Setup": l, "Runs": len(df), "Success %": round(100 * ok.mean()),
                     "People reached": f"{ssum(df, 'reached')}/{ssum(df, 'people')}",
                     "Time to reach all (s)": round(float(t[ok].mean()), 1) if ok.any() else None,
                     "Ghost rescues": ssum(df, "ghost_rescues"), "Robot-caused collisions": ssum(df, "robot_collisions"),
                     "Path (m)": round(float(pd.to_numeric(df["path_m"], errors="coerce").mean()), 1)})
    return pd.DataFrame(rows)


def paired(a, b):
    A, B = load_batch(a), load_batch(b)
    if A is None or B is None:
        return None
    t = lambda r: float(r["time_all"]) if str(r["success"]) == "True" else 180.0
    A, B = A.set_index("seed"), B.set_index("seed")
    seeds = sorted(set(A.index) & set(B.index))
    d = np.array([t(B.loc[s]) - t(A.loc[s]) for s in seeds])
    boot = [np.random.default_rng(k).choice(d, len(d)).mean() for k in range(3000)]
    return len(seeds), d.mean(), np.percentile(boot, 2.5), np.percentile(boot, 97.5), (d < -0.5).sum(), (d > 0.5).sum()


# ---------------- pages ----------------
page = st.sidebar.radio("Section", ["Overview", "Perception", "Coordination", "Decisions", "Robustness", "Mission replay"])
st.sidebar.markdown("---")
st.sidebar.caption("Multi-Agent Depth-Aware Autonomous Decision System Using Infrared Vision · Amrita Vishwa Vidyapeetham")

if page == "Overview":
    st.title("Multi-agent thermal search and rescue")
    st.write("Night-time rescue at a city intersection in fog. Three agents (a fixed pole camera, a ground robot and a "
             "drone) see only through **thermal cameras**, share what they see over radio, build one shared map of "
             "people, and decide who goes where.")
    st.graphviz_chart("""digraph { rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#eef3fb", fontname=Helvetica];
        a [label="Thermal cameras\\n(3 agents)"]; b [label="Detection\\nYOLOv8s"]; c [label="Tracking\\nBoT-SORT"];
        d [label="Distance\\ndepth net + ground plane"]; e [label="Map position\\n+ uncertainty"]; f [label="Radio\\nsharing"];
        g [label="Kalman fusion\\nshared map"]; h [label="Decisions\\nrule-based / MAPPO"]; i [label="Robots act\\nsafely"];
        a->b->c->d->e->f->g->h->i; }""")
    c = st.columns(5)
    c[0].metric("Detection mAP50 (sim test)", "0.967", "+0.375 after fine-tuning")
    c[1].metric("Depth RMSE", "1.89 m")
    c[2].metric("Shared map error", "0.08 m", "100% coverage")
    c[3].metric("Mission success (20 runs)", "95%", "0 robot collisions (RL v2)")
    c[4].metric("Night + fog 0.09 recall", "90% thermal", "vs 9% RGB")
    st.caption("All numbers are read from / documented in outputs/results/.")

elif page == "Perception":
    st.title("Perception: detection, tracking, depth")
    show_md("detection_results.md")
    st.markdown("---")
    show_md("tracking_replay.md")
    gif = f"{ROOT}/outputs/tracking/pole_cam_tracking.gif"
    if os.path.exists(gif):
        st.image(gif, caption="Pole camera: detection + tracking IDs", width=480)
    st.markdown("---")
    show_md("depth_results.md")

elif page == "Coordination":
    st.title("Coordination: localisation, fusion, communication")
    show_md("localization_results.md")
    show_md("world_localization_results.md")
    show_md("fusion_results.md")
    show_md("live_fusion_results.md")
    st.subheader("Live missions vs message loss")
    df = summary([l for l in ("rule_v5", "rule_loss50", "rule_loss100") if l in batches()])
    if not df.empty:
        df["Message loss"] = df["Setup"].map({"rule_v5": "0%", "rule_loss50": "50%", "rule_loss100": "100%"})
        st.bar_chart(df.set_index("Message loss")[["Success %"]])
        st.dataframe(df, hide_index=True)

elif page == "Decisions":
    st.title("Decisions: rule-based vs MAPPO")
    all_b = batches()
    default = [b for b in ("rule_v5", "rl_v1", "rl_v2", "rule_nopole_v3") if b in all_b]
    chosen = st.multiselect("Batches to compare", all_b, default=default)
    st.dataframe(summary(chosen), hide_index=True)
    st.subheader("Paired comparison (same seeds)")
    col = st.columns(2)
    a = col[0].selectbox("Baseline", all_b, index=all_b.index("rule_v5") if "rule_v5" in all_b else 0)
    b = col[1].selectbox("Compared", all_b, index=all_b.index("rl_v2") if "rl_v2" in all_b else 0)
    r = paired(a, b)
    if r:
        n, m, lo, hi, faster, slower = r
        st.write(f"**{n} paired seeds**: time difference ({b} − {a}) = **{m:+.2f} s**, 95% CI [{lo:+.2f}, {hi:+.2f}] · "
                 f"{b} faster on {faster}, slower on {slower} seeds (failures counted as 180 s).")
        st.caption("If the interval contains 0, the difference is not statistically significant.")
    show_md("rule_based_results.md", "Rule-based development history")

elif page == "Robustness":
    st.title("Robustness: fog and thermal vs RGB")
    fogs = sorted([b for b in batches() if b.startswith("fog_")], key=lambda x: float(x[4:]))
    if fogs:
        df = summary(fogs)
        df["Fog"] = df["Setup"].str[4:]
        st.subheader("Missions vs fog")
        st.line_chart(df.set_index("Fog")[["Success %"]])
        st.dataframe(df, hide_index=True)
    t = md_table("fog_detection_curve.md")
    if t is not None:
        st.subheader("Thermal detection vs fog")
        t = t.rename(columns={t.columns[0]: "Fog"})
        for c in t.columns[2:]:
            t[c] = pd.to_numeric(t[c], errors="coerce")
        st.line_chart(t.set_index("Fog")[list(t.columns[2:])])
    st.markdown("---")
    show_md("rgb_vs_thermal.md")
    img = f"{RES}/rgb_vs_thermal_example.png"
    if os.path.exists(img):
        st.image(img, caption="Blue = true people, green = detections")

else:
    st.title("Mission replay")
    paths = sorted(glob.glob(f"{ROOT}/outputs/missions/mission_*.json"), key=os.path.getmtime, reverse=True)
    filt = st.text_input("Filter missions (e.g. rl_v2_s5, rule_v5, fog_0.15)", "rl_v2")
    paths = [p for p in paths if filt in os.path.basename(p)] or paths
    path = st.selectbox("Mission", paths, format_func=os.path.basename)
    ts = lambda p: re.search(r"(\d{8}_\d{6})", os.path.basename(p)).group(1)

    @st.cache_data
    def load(p):
        m = json.load(open(p))
        frames = [json.loads(l) for l in open(m["perception_log"]) if l.strip()]
        truths = [q for q in sorted(glob.glob(f"{ROOT}/outputs/live_logs/truth_*.jsonl")) if ts(q) <= ts(m["perception_log"])]
        truth = [json.loads(l) for l in open(truths[-1]) if l.strip()] if truths else []
        return m, frames, truth

    try:
        m, frames, truth = load(path)
    except Exception as e:
        st.error(f"Could not load this mission's logs: {e}")
        st.stop()
    st.write(f"**Policy:** {m.get('policy')} · **seed:** {m.get('seed')} · **fog:** {(m.get('conditions') or {}).get('fog')} · "
             f"**pole camera:** {m.get('pole', True)} · **ended:** {m['end_t']:.0f} s ({m['reason']}) · "
             f"**rescues claimed:** {len(m['rescues'])}")
    t = st.slider("Time (s)", 0.0, float(m["end_t"]), float(min(30.0, m["end_t"])), 0.5)
    tt = np.array([r["t"] for r in truth])
    r = truth[int(np.argmin(np.abs(tt - t)))]
    people = [o for o in r["objects"] if o["cls"] == 0]
    path_xy = np.array([next(o["pos"][:2] for o in q["objects"] if o["name"] == "ugv") for q in truth if q["t"] <= t])
    ft = np.array([f["t"] for f in frames])
    fr = frames[int(np.argmin(np.abs(ft - t)))]
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.add_patch(plt.Rectangle((36, -54), 18, 18, fill=False, ls=":", color="gray"))
    for o in people:
        walking = o["name"] in ("pedestrian(1)", "pedestrian(2)")
        ax.scatter(*o["pos"][:2], s=120, c="tab:blue", marker="^" if walking else "o", zorder=3)
    for q in fr.get("fused", []):
        ax.scatter(*q["xy"], s=160, facecolors="none", edgecolors="tab:orange", linewidths=2, zorder=4)
    if len(path_xy):
        ax.plot(path_xy[:, 0], path_xy[:, 1], color="gray", lw=1.5)
        ax.scatter(*path_xy[-1], s=200, c="black", marker="s", zorder=5)
    for c in m["rescues"]:
        if c["t"] <= t:
            ax.scatter(*c["xy"], s=260, c="tab:green", marker="*", zorder=6)
    ax.scatter([], [], c="tab:blue", marker="o", label="person (standing)")
    ax.scatter([], [], c="tab:blue", marker="^", label="person (walking)")
    ax.scatter([], [], facecolors="none", edgecolors="tab:orange", label="robot's shared map")
    ax.scatter([], [], c="black", marker="s", label="ground robot (path in grey)")
    ax.scatter([], [], c="tab:green", marker="*", label="rescue claimed")
    ax.set_xlim(30, 60); ax.set_ylim(-60, -10); ax.set_aspect("equal"); ax.grid(alpha=0.3)
    ax.set_xlabel("east (m)"); ax.set_ylabel("north (m)"); ax.legend(loc="upper left", fontsize=8)
    ax.set_title(f"t = {t:.1f} s")
    c1, c2 = st.columns([2, 1])
    c1.pyplot(fig)
    c2.metric("People on robot's map", len(fr.get("fused", [])))
    c2.metric("Rescues claimed so far", sum(c["t"] <= t for c in m["rescues"]))
    c2.metric("Messages received", fr.get("comm", {}).get("received", 0))
