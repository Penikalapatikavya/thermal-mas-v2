"""Thermal multi-agent rescue: interactive results dashboard.   Run: streamlit run dashboard/app.py"""
import glob, json, os, re
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = os.path.expanduser("~/thermal-mas-v2")
RES = f"{ROOT}/outputs/results"
HOT = ["#1d4ed8", "#0ea5e9", "#14b8a6", "#6366f1", "#94a3b8", "#0f766e", "#10b981"]
st.set_page_config(page_title="Thermal multi-agent rescue", page_icon="🔥", layout="wide")
st.markdown("""<style>
.hero {background: linear-gradient(120deg,#1e3a8a 0%,#1d4ed8 40%,#0ea5e9 75%,#14b8a6 100%);
       padding: 26px 30px; border-radius: 16px; margin-bottom: 18px; box-shadow: 0 8px 24px rgba(29,78,216,.18);}
.hero h1 {color: white; margin: 0; font-size: 2.1rem;} .hero p {color: #e0f2fe; margin: 6px 0 0 0; font-size: 1.05rem;}
.card {background: #ffffff; border-radius: 14px; padding: 16px 18px; border-top: 4px solid var(--c);
       box-shadow: 0 2px 10px rgba(15,23,42,.08); height: 100%;}
.card .v {font-size: 1.9rem; font-weight: 800; color: var(--c);} .card .l {color: #334155; font-size: .9rem;}
.card .s {color: #64748b; font-size: .78rem; margin-top: 4px;}
[data-testid="stSidebar"] {background: linear-gradient(180deg,#0f172a 0%,#1e3a8a 100%);}
[data-testid="stSidebar"] * {color: #e2e8f0 !important;}
</style>""", unsafe_allow_html=True)


def hero(title, sub):
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{sub}</p></div>', unsafe_allow_html=True)


def cards(items):
    cols = st.columns(len(items))
    for col, (val, label, sub, color) in zip(cols, items):
        col.markdown(f'<div class="card" style="--c:{color}"><div class="v">{val}</div><div class="l">{label}</div>'
                     f'<div class="s">{sub}</div></div>', unsafe_allow_html=True)


def style(fig, h=420):
    fig.update_layout(template="plotly_white", height=h, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#ffffff",
                      margin=dict(l=40, r=20, t=50, b=40), legend=dict(bgcolor="rgba(0,0,0,0)"), font=dict(size=13))
    return fig


def show_md(name):
    p = f"{RES}/{name}"
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
    body = [[c.strip() for c in r.strip("|").split("|")] for r in rows[2:] if not set(r) <= set("|-: ")]
    body = [b for b in body if len(b) == len(head)]
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
    df["ok"] = df["success"].astype(str) == "True"
    df["time"] = np.where(df["ok"], pd.to_numeric(df.get("time_all"), errors="coerce"), 180.0)
    return df


def ssum(df, col):
    return int(np.nansum(pd.to_numeric(df[col], errors="coerce"))) if col in df else 0


def summary(labels):
    rows = []
    for l in labels:
        df = load_batch(l)
        if df is None or df.empty:
            continue
        ok = df["ok"]
        rows.append({"Setup": l, "Runs": len(df), "Success %": round(100 * ok.mean()),
                     "People reached": f"{ssum(df, 'reached')}/{ssum(df, 'people')}",
                     "Time to reach all (s)": round(float(df.loc[ok, "time"].mean()), 1) if ok.any() else None,
                     "Ghost rescues": ssum(df, "ghost_rescues"), "Robot-caused collisions": ssum(df, "robot_collisions"),
                     "Path (m)": round(float(pd.to_numeric(df["path_m"], errors="coerce").mean()), 1)})
    return pd.DataFrame(rows)


# ---------------- navigation ----------------
st.sidebar.markdown("## 🔥 Thermal rescue")
page = st.sidebar.radio("Section", ["🏠 Overview", "👁️ Perception", "📡 Coordination", "🧠 Decisions",
                                    "🌫️ Robustness", "🎬 Mission replay"], label_visibility="collapsed")
st.sidebar.markdown("---")
st.sidebar.caption("Multi-Agent Depth-Aware Autonomous Decision System Using Infrared Vision\n\nAmrita Vishwa Vidyapeetham")

# ---------------- pages ----------------
if page.endswith("Overview"):
    hero("Multi-agent thermal search & rescue",
         "Night. Fog. A city intersection. Three thermal-sighted robots find and reach stranded people together.")
    cards([("0.967", "Detection mAP50", "sim test, +0.375 after fine-tuning", HOT[0]),
           ("1.89 m", "Depth RMSE", "monocular thermal U-Net", HOT[1]),
           ("0.08 m", "Shared-map error", "100% coverage, 3 agents fused", HOT[2]),
           ("95%", "Mission success", "20 seeded Webots missions", HOT[3]),
           ("90% vs 9%", "Thermal vs RGB recall", "night, fog 0.09", HOT[5])])
    st.write("")
    st.graphviz_chart("""digraph { rankdir=LR; bgcolor="transparent";
        node [shape=box, style="rounded,filled", fontname=Helvetica, fontcolor=white, color="#00000000"];
        edge [color="#14b8a6", penwidth=2];
        a [label="Thermal cameras\\n(3 agents)", fillcolor="#1e3a8a"]; b [label="Detection\\nYOLOv8s", fillcolor="#1d4ed8"];
        c [label="Tracking\\nBoT-SORT", fillcolor="#2563eb"]; d [label="Distance\\ndepth + ground plane", fillcolor="#0284c7"];
        e [label="Map position\\n+ uncertainty", fillcolor="#0ea5e9"]; f [label="Radio\\nsharing", fillcolor="#06b6d4"];
        g [label="Kalman fusion\\nshared map", fillcolor="#14b8a6"]; h [label="Decisions\\nrule / MAPPO", fillcolor="#0d9488"];
        i [label="Robots act\\nsafely", fillcolor="#0f766e"]; a->b->c->d->e->f->g->h->i; }""")
    t1, t2, t3 = st.tabs(["🎯 The scenario", "🤖 The agents", "❓ Research questions"])
    t1.markdown("A night-time rescue at a city intersection. **4 people** are stranded (2 standing, 2 walking), a parked car "
                "with a warm engine acts as a distractor, and fog varies from clear to very heavy. The cameras see only "
                "**heat**: no daylight, no LiDAR.")
    t2.markdown("- **Pole camera**: fixed, 10 m high, sees the whole intersection\n"
                "- **Ground robot** (Pioneer 3-DX): drives to people and reaches them safely\n"
                "- **Drone** (Mavic 2 Pro): flies over the area and looks where others can't")
    t3.markdown("- **RQ1**: can thermal perception detect, track and localise people reliably?\n"
                "- **RQ2**: does sharing beliefs between agents beat working alone, and at what communication cost?\n"
                "- **RQ3**: how robust is the system in low visibility (night, fog)?")

elif page.endswith("Perception"):
    hero("👁️ Perception", "Seeing people in heat: detection, tracking and distance")
    t1, t2, t3 = st.tabs(["Detection", "Tracking", "Depth"])
    with t1:
        cards([("0.592 → 0.967", "mAP50 on simulation", "after fine-tuning", HOT[0]),
               ("0.740 → 0.740", "mAP50 on real FLIR", "no forgetting", HOT[5])])
        st.write("")
        fig = go.Figure([go.Bar(name="Before fine-tuning", x=["All", "Person", "Car"], y=[0.592, 0.814, 0.371], marker_color=HOT[4]),
                         go.Bar(name="After fine-tuning", x=["All", "Person", "Car"], y=[0.967, 0.981, 0.953], marker_color=HOT[1])])
        fig.update_layout(barmode="group", title="Simulation test set, mAP50", yaxis_range=[0, 1])
        st.plotly_chart(style(fig), use_container_width=True)
        with st.expander("Full detection results"):
            show_md("detection_results.md")
    with t2:
        gif = f"{ROOT}/outputs/tracking/pole_cam_tracking.gif"
        c1, c2 = st.columns([1, 1])
        if os.path.exists(gif):
            c1.image(gif, caption="Pole camera: detection + track IDs")
        fig = go.Figure([go.Bar(name="ByteTrack", x=["Recall", "MOTA"], y=[0.262, 0.227], marker_color=HOT[3]),
                         go.Bar(name="BoT-SORT (camera-motion compensation)", x=["Recall", "MOTA"], y=[0.715, 0.633],
                                marker_color=HOT[0])])
        fig.update_layout(barmode="group", title="Drone (moving camera): tracker comparison", yaxis_range=[0, 1])
        c2.plotly_chart(style(fig, 380), use_container_width=True)
        with st.expander("Full tracking results"):
            show_md("tracking_replay.md")
    with t3:
        t = md_table("depth_results.md")
        if t is not None and "Method" in t.columns:
            t["RMSE (m)"] = pd.to_numeric(t["RMSE (m)"], errors="coerce")
            fig = go.Figure()
            for i, (meth, g) in enumerate(t.groupby("Method")):
                fig.add_bar(name=meth, x=g["Agent"], y=g["RMSE (m)"], marker_color=[HOT[1], HOT[4]][i % 2])
            fig.update_layout(barmode="group", title="Depth error per agent (lower is better)", yaxis_title="RMSE (m)")
            st.plotly_chart(style(fig), use_container_width=True)
        with st.expander("Full depth results"):
            show_md("depth_results.md")

elif page.endswith("Coordination"):
    hero("📡 Coordination", "From pixels to one shared map, over an imperfect radio")
    cards([("0.72 m", "Best single agent", "pole camera alone", HOT[4]),
           ("0.08 m", "All three fused", "crossing lines of sight", HOT[0]),
           ("4 IDs", "for 4 people", "map-level tracking, 0 ID switches", HOT[5])])
    st.write("")
    t1, t2, t3 = st.tabs(["Message loss (live missions)", "Fusion tables", "Localisation"])
    with t1:
        df = summary([l for l in ("rule_v5", "rule_loss50", "rule_loss100") if l in batches()])
        if not df.empty:
            df["Loss"] = df["Setup"].map({"rule_v5": "0%", "rule_loss50": "50%", "rule_loss100": "100%"})
            fig = go.Figure()
            fig.add_bar(x=df["Loss"], y=df["Success %"], name="Success %", marker_color=[HOT[0], HOT[1], HOT[2]],
                        text=df["Success %"], textposition="outside")
            fig.add_scatter(x=df["Loss"], y=df["Robot-caused collisions"], name="Robot-caused collisions",
                            yaxis="y2", mode="lines+markers", line=dict(color=HOT[5], width=4), marker=dict(size=14))
            fig.update_layout(title="Mission success and safety vs message loss", yaxis=dict(title="Success %", range=[0, 110]),
                              yaxis2=dict(title="Collisions", overlaying="y", side="right"), xaxis_title="Messages lost")
            st.plotly_chart(style(fig, 460), use_container_width=True)
            st.dataframe(df.drop(columns="Loss"), hide_index=True, use_container_width=True)
    with t2:
        show_md("fusion_results.md")
        show_md("live_fusion_results.md")
    with t3:
        show_md("localization_results.md")
        show_md("world_localization_results.md")

elif page.endswith("Decisions"):
    hero("🧠 Decisions", "Hand-written rules vs a trained multi-agent team (MAPPO)")
    all_b = batches()
    t1, t2, t3 = st.tabs(["Compare batches", "Rule vs RL, mission by mission", "History"])
    with t1:
        default = [b for b in ("rule_v5", "rl_v1", "rl_v2", "rule_nopole_v3") if b in all_b]
        chosen = st.multiselect("Batches", all_b, default=default)
        df = summary(chosen)
        if not df.empty:
            fig = go.Figure()
            for i, l in enumerate(chosen):
                b = load_batch(l)
                if b is not None:
                    fig.add_box(y=b["time"], name=l, boxpoints="all", jitter=0.4, pointpos=0,
                                marker=dict(color=HOT[i % len(HOT)], size=8), line=dict(color=HOT[i % len(HOT)]),
                                hovertext=[f"seed {s}" for s in b["seed"]])
            fig.update_layout(title="Time to reach all people, every mission (failures shown at 180 s)",
                              yaxis_title="seconds", showlegend=False)
            st.plotly_chart(style(fig, 460), use_container_width=True)
            st.dataframe(df, hide_index=True, use_container_width=True)
    with t2:
        c = st.columns(2)
        a = c[0].selectbox("Baseline", all_b, index=all_b.index("rule_v5") if "rule_v5" in all_b else 0)
        bb = c[1].selectbox("Compared", all_b, index=all_b.index("rl_v2") if "rl_v2" in all_b else 0)
        A, B = load_batch(a), load_batch(bb)
        if A is not None and B is not None:
            m = A.merge(B, on="seed", suffixes=("_a", "_b"))
            d = m["time_b"] - m["time_a"]
            boot = [np.random.default_rng(k).choice(d, len(d)).mean() for k in range(3000)]
            lo, hi = np.percentile(boot, [2.5, 97.5])
            cards([(f"{d.mean():+.1f} s", f"Mean difference ({bb} − {a})", f"95% CI [{lo:+.1f}, {hi:+.1f}] s", HOT[1]),
                   (f"{(d < -0.5).sum()}", f"Seeds where {bb} was faster", "", HOT[6]),
                   (f"{(d > 0.5).sum()}", f"Seeds where {bb} was slower", "", HOT[2])])
            fig = go.Figure()
            lim = [70, 185]
            fig.add_scatter(x=lim, y=lim, mode="lines", line=dict(color="#666", dash="dash"), name="equal")
            fig.add_scatter(x=m["time_a"], y=m["time_b"], mode="markers+text", text=m["seed"], textposition="top center",
                            marker=dict(size=14, color=d, colorscale="RdYlGn_r", cmin=-15, cmax=15, showscale=True,
                                        colorbar=dict(title="Δ s")), name="mission (seed)")
            fig.update_layout(title=f"Each dot is one mission: below the line = {bb} faster",
                              xaxis_title=f"{a} time (s)", yaxis_title=f"{bb} time (s)", xaxis_range=lim, yaxis_range=lim)
            st.plotly_chart(style(fig, 520), use_container_width=True)
            st.caption("If the confidence interval contains 0, the difference is not statistically significant.")
    with t3:
        show_md("rule_based_results.md")

elif page.endswith("Robustness"):
    hero("🌫️ Robustness", "What happens when the night gets darker and the fog gets thicker")
    t1, t2 = st.tabs(["Fog", "Thermal vs RGB"])
    with t1:
        fogs = sorted([b for b in batches() if b.startswith("fog_")], key=lambda x: float(x[4:]))
        fig = go.Figure()
        if fogs:
            df = summary(fogs)
            df["Fog"] = df["Setup"].str[4:].astype(float)
            fig.add_scatter(x=df["Fog"], y=df["Success %"] / 100, name="Mission success", mode="lines+markers",
                            line=dict(color=HOT[1], width=5), marker=dict(size=14))
        t = md_table("fog_detection_curve.md")
        if t is not None:
            fig.add_scatter(x=pd.to_numeric(t.iloc[:, 0], errors="coerce"), y=pd.to_numeric(t.iloc[:, 2], errors="coerce"),
                            name="Detection recall", mode="lines+markers", line=dict(color=HOT[5], width=4, dash="dot"),
                            marker=dict(size=11))
        fig.add_vrect(x0=0, x1=0.04, fillcolor=HOT[6], opacity=0.12, line_width=0, annotation_text="training range")
        fig.update_layout(title="Detection and mission success vs fog", xaxis_title="fog attenuation per metre",
                          yaxis=dict(range=[0, 1.05], tickformat=".0%"))
        st.plotly_chart(style(fig, 480), use_container_width=True)
        if fogs:
            st.dataframe(df.drop(columns="Fog"), hide_index=True, use_container_width=True)
    with t2:
        t = md_table("rgb_vs_thermal.md")
        if t is not None:
            light = st.radio("Lighting", ["Day", "Night"], horizontal=True, index=1)
            rows = t[t.iloc[:, 0].str.startswith(light)]
            fogx = rows.iloc[:, 0].str.extract(r"fog ([\d.]+)")[0]
            rgb = rows.iloc[:, 1].str.split("/").str[0].astype(float)
            th = rows.iloc[:, 2].str.split("/").str[0].astype(float)
            fig = go.Figure([go.Bar(name="RGB camera (COCO YOLOv8s)", x=fogx, y=rgb, marker_color="#8d99ae",
                                    text=[f"{v:.0%}" for v in rgb], textposition="outside"),
                             go.Bar(name="Thermal camera (ours)", x=fogx, y=th, marker_color=HOT[1],
                                    text=[f"{v:.0%}" for v in th], textposition="outside")])
            fig.update_layout(barmode="group", title=f"Person detection recall, {light.lower()}",
                              xaxis_title="fog attenuation per metre", yaxis=dict(range=[0, 1.1], tickformat=".0%"))
            st.plotly_chart(style(fig, 460), use_container_width=True)
        img = f"{RES}/rgb_vs_thermal_example.png"
        if os.path.exists(img):
            st.image(img, caption="Same moment: RGB by day · RGB at night in fog · thermal in fog (blue = true people, green = detected)")
        with st.expander("Full table and method"):
            show_md("rgb_vs_thermal.md")

else:
    hero("🎬 Mission replay", "Press ▶ to watch a recorded mission: people, robot, shared map and rescues")
    paths = sorted(glob.glob(f"{ROOT}/outputs/missions/mission_*.json"), key=os.path.getmtime, reverse=True)
    c = st.columns([1, 2])
    filt = c[0].text_input("Filter (e.g. rl_v2_s5, rule_v5, fog_0.15)", "rl_v2")
    sel = [p for p in paths if filt in os.path.basename(p)] or paths
    path = c[1].selectbox("Mission", sel, format_func=os.path.basename)
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
    cards([(str(m.get("policy")), "Policy", f"seed {m.get('seed')}", HOT[1]),
           (str((m.get("conditions") or {}).get("fog")), "Fog", "attenuation per metre", HOT[5]),
           (f"{m['end_t']:.0f} s", "Mission end", m["reason"], HOT[0]),
           (str(len(m["rescues"])), "Rescues claimed", f"pole camera: {m.get('pole', True)}", HOT[6])])
    st.write("")
    tt = np.array([r["t"] for r in truth])
    ft = np.array([f["t"] for f in frames])
    times = np.arange(1.0, float(m["end_t"]) + 0.01, 1.0)

    def frame_data(t):
        r = truth[int(np.argmin(np.abs(tt - t)))]
        people = [o for o in r["objects"] if o["cls"] == 0]
        stand = [o["pos"][:2] for o in people if o["name"] not in ("pedestrian(1)", "pedestrian(2)")]
        walk = [o["pos"][:2] for o in people if o["name"] in ("pedestrian(1)", "pedestrian(2)")]
        trail = [next(o["pos"][:2] for o in q["objects"] if o["name"] == "ugv") for q in truth if q["t"] <= t]
        fused = frames[int(np.argmin(np.abs(ft - t)))].get("fused", [])
        resc = [c["xy"] for c in m["rescues"] if c["t"] <= t]
        xy = lambda L: ([p[0] for p in L], [p[1] for p in L])
        return [
            go.Scatter(x=xy(trail)[0], y=xy(trail)[1], mode="lines", line=dict(color=HOT[1], width=3), name="robot trail"),
            go.Scatter(x=xy(stand)[0], y=xy(stand)[1], mode="markers", marker=dict(size=16, color=HOT[0], symbol="circle",
                       line=dict(color="white", width=1)), name="person (standing)"),
            go.Scatter(x=xy(walk)[0], y=xy(walk)[1], mode="markers", marker=dict(size=16, color=HOT[3], symbol="triangle-up",
                       line=dict(color="white", width=1)), name="person (walking)"),
            go.Scatter(x=xy([q["xy"] for q in fused])[0], y=xy([q["xy"] for q in fused])[1], mode="markers",
                       marker=dict(size=26, color="rgba(0,0,0,0)", line=dict(color=HOT[5], width=3)), name="robot's shared map"),
            go.Scatter(x=[trail[-1][0]] if trail else [], y=[trail[-1][1]] if trail else [], mode="markers",
                       marker=dict(size=20, color="#0f172a", symbol="square", line=dict(color=HOT[1], width=3)), name="ground robot"),
            go.Scatter(x=xy(resc)[0], y=xy(resc)[1], mode="markers", marker=dict(size=26, color="#f59e0b", symbol="star"),
                       name="rescue claimed"),
        ]

    fig = go.Figure(data=frame_data(times[0]),
                    frames=[go.Frame(data=frame_data(t), name=f"{t:.0f}") for t in times])
    fig.add_shape(type="rect", x0=36, y0=-54, x1=54, y1=-36, line=dict(color="#555", dash="dot"))
    fig.update_layout(
        xaxis=dict(range=[30, 60], title="east (m)", gridcolor="#e5e7eb"), yaxis=dict(range=[-60, -10], title="north (m)",
        scaleanchor="x", gridcolor="#e5e7eb"),
        updatemenus=[dict(type="buttons", x=0, y=-0.08, xanchor="left", direction="left", buttons=[
            dict(label="▶ Play", method="animate", args=[None, dict(frame=dict(duration=250, redraw=True), fromcurrent=True)]),
            dict(label="⏸ Pause", method="animate", args=[[None], dict(mode="immediate", frame=dict(duration=0))])])],
        sliders=[dict(active=0, y=-0.02, currentvalue=dict(prefix="t = ", suffix=" s"),
                      steps=[dict(method="animate", label=f"{t:.0f}", args=[[f"{t:.0f}"], dict(mode="immediate",
                              frame=dict(duration=0, redraw=True))]) for t in times])])
    st.plotly_chart(style(fig, 720), use_container_width=True)
    st.caption("Dotted square: where standing people can be placed. Orange rings = where the robot's shared map believes people are.")
