"""Thermal multi-agent rescue: results dashboard (polished).   Run: streamlit run dashboard/app.py"""
import glob, json, os, re
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = os.path.expanduser("~/thermal-mas-v2")
RES = f"{ROOT}/outputs/results"
C = {"navy": "#0f172a", "blue": "#1d4ed8", "sky": "#0ea5e9", "teal": "#14b8a6", "indigo": "#6366f1",
     "slate": "#94a3b8", "dteal": "#0f766e", "green": "#10b981", "amber": "#f59e0b", "rose": "#f43f5e"}
SEQ = [C["blue"], C["teal"], C["sky"], C["indigo"], C["dteal"], C["amber"], C["rose"]]
st.set_page_config(page_title="Thermal multi-agent rescue", page_icon="🛰️", layout="wide")

st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stMarkdown, button, input {font-family: 'Inter', sans-serif !important;}
.stApp {background: radial-gradient(1200px 600px at 100% 0%, #e0f2fe 0%, rgba(224,242,254,0) 60%),
                    radial-gradient(900px 500px at 0% 100%, #ccfbf1 0%, rgba(204,251,241,0) 60%), #f6f9fc;}
.block-container {padding-top: 1.6rem; max-width: 1300px;}
.hero {position: relative; overflow: hidden; border-radius: 22px; padding: 34px 38px; margin-bottom: 22px;
       background: linear-gradient(120deg, #0b1f5c 0%, #1d4ed8 45%, #0ea5e9 78%, #14b8a6 100%);
       box-shadow: 0 18px 40px rgba(29,78,216,.25);}
.hero:before, .hero:after {content: ""; position: absolute; border-radius: 50%; background: rgba(255,255,255,.10);}
.hero:before {width: 320px; height: 320px; right: -80px; top: -120px;}
.hero:after {width: 200px; height: 200px; right: 180px; bottom: -110px; background: rgba(255,255,255,.07);}
.hero h1 {color: #fff; margin: 0; font-size: 2.3rem; font-weight: 800; letter-spacing: -.02em;}
.hero p {color: #e0f2fe; margin: 8px 0 0 0; font-size: 1.08rem; max-width: 760px;}
.pill {display: inline-block; margin: 14px 8px 0 0; padding: 5px 12px; border-radius: 999px; font-size: .78rem;
       font-weight: 600; color: #fff; background: rgba(255,255,255,.16); border: 1px solid rgba(255,255,255,.25);}
.card {background: #fff; border-radius: 18px; padding: 18px 20px; height: 100%; border: 1px solid #e6edf6;
       box-shadow: 0 4px 16px rgba(15,23,42,.06); transition: transform .15s ease, box-shadow .15s ease;}
.card:hover {transform: translateY(-3px); box-shadow: 0 12px 28px rgba(15,23,42,.10);}
.card .ic {width: 42px; height: 42px; border-radius: 12px; display: flex; align-items: center; justify-content: center;
           font-size: 1.25rem; background: var(--bg); margin-bottom: 10px;}
.card .v {font-size: 1.85rem; font-weight: 800; color: var(--c); letter-spacing: -.02em; line-height: 1.1;}
.card .l {color: #1e293b; font-weight: 600; font-size: .92rem; margin-top: 4px;}
.card .s {color: #64748b; font-size: .8rem; margin-top: 3px;}
.sec {display: flex; align-items: center; gap: 10px; margin: 26px 0 10px 0;}
.sec .bar {width: 6px; height: 26px; border-radius: 4px; background: linear-gradient(180deg, #1d4ed8, #14b8a6);}
.sec h3 {margin: 0; font-weight: 700; color: #0f172a;}
.flow {display: flex; flex-wrap: wrap; gap: 10px; align-items: center;}
.step {background: #fff; border: 1px solid #e6edf6; border-radius: 14px; padding: 10px 14px; display: flex;
       gap: 10px; align-items: center; box-shadow: 0 2px 8px rgba(15,23,42,.05);}
.step .n {width: 26px; height: 26px; border-radius: 50%; color: #fff; font-weight: 700; font-size: .8rem; display: flex;
          align-items: center; justify-content: center; background: linear-gradient(135deg, #1d4ed8, #14b8a6);}
.step .t {font-weight: 600; color: #0f172a; font-size: .88rem;} .step .d {color: #64748b; font-size: .74rem;}
.arrow {color: #14b8a6; font-weight: 800;}
.deliv {background: #fff; border: 1px solid #e6edf6; border-radius: 16px; padding: 14px 16px; height: 100%;
        box-shadow: 0 2px 10px rgba(15,23,42,.05);}
.deliv .h {display: flex; justify-content: space-between; align-items: center; font-weight: 700; color: #0f172a;}
.chip {font-size: .7rem; font-weight: 700; padding: 3px 9px; border-radius: 999px; background: #dcfce7; color: #166534;}
.chip.wait {background: #fef3c7; color: #92400e;}
.deliv .b {color: #475569; font-size: .84rem; margin-top: 6px;}
.stTabs [data-baseweb="tab-list"] {gap: 8px;}
.stTabs [data-baseweb="tab"] {background: #fff; border: 1px solid #e6edf6; border-radius: 999px; padding: 6px 16px; height: auto;}
.stTabs [aria-selected="true"] {background: linear-gradient(90deg, #1d4ed8, #14b8a6) !important; color: #fff !important; border: none;}
.stTabs [aria-selected="true"] p {color: #fff !important;}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {display: none;}
[data-testid="stSidebar"] {background: linear-gradient(180deg, #0b1220 0%, #12245e 100%);}
[data-testid="stSidebar"] * {color: #e2e8f0 !important;}
.badge {background: linear-gradient(135deg, #1d4ed8, #14b8a6); border-radius: 16px; padding: 16px; margin-bottom: 14px;}
.badge b {font-size: 1.05rem;} .badge span {font-size: .78rem; opacity: .9;}
[data-testid="stDataFrame"] {border-radius: 14px; overflow: hidden; border: 1px solid #e6edf6;}
</style>""", unsafe_allow_html=True)


# ---------------- UI helpers ----------------
def hero(title, sub, pills=()):
    p = "".join(f'<span class="pill">{x}</span>' for x in pills)
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{sub}</p>{p}</div>', unsafe_allow_html=True)


def cards(items):
    cols = st.columns(len(items))
    for col, (icon, val, label, sub, color) in zip(cols, items):
        col.markdown(f'<div class="card" style="--c:{color};--bg:{color}1f"><div class="ic">{icon}</div>'
                     f'<div class="v">{val}</div><div class="l">{label}</div><div class="s">{sub}</div></div>',
                     unsafe_allow_html=True)


def section(title):
    st.markdown(f'<div class="sec"><div class="bar"></div><h3>{title}</h3></div>', unsafe_allow_html=True)


def style(fig, h=420):
    fig.update_layout(template="plotly_white", height=h, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#ffffff",
                      font=dict(family="Inter, sans-serif", size=13, color="#1e293b"), colorway=SEQ, barcornerradius=8,
                      margin=dict(l=50, r=30, t=60, b=50), title_font=dict(size=16, color="#0f172a"),
                      legend=dict(bgcolor="rgba(255,255,255,0.8)", bordercolor="#e6edf6", borderwidth=1),
                      xaxis=dict(gridcolor="#eef2f7", zeroline=False), yaxis=dict(gridcolor="#eef2f7", zeroline=False))
    return fig


def show_md(name):
    p = f"{RES}/{name}"
    st.markdown(open(p).read()) if os.path.exists(p) else st.info(f"{name} not found in outputs/results")


def md_table(name):
    p = f"{RES}/{name}"
    if not os.path.exists(p):
        return None
    rows = [l.strip() for l in open(p) if l.strip().startswith("|")]
    if len(rows) < 3:
        return None
    head = [c.strip() for c in rows[0].strip("|").split("|")]
    body = [[c.strip() for c in r.strip("|").split("|")] for r in rows[2:] if not set(r) <= set("|-: ")]
    return pd.DataFrame([b for b in body if len(b) == len(head)], columns=head)


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


# ---------------- sidebar ----------------
st.sidebar.markdown('<div class="badge"><b>🛰️ Thermal Rescue MAS</b><br><span>Multi-agent · depth-aware · infrared</span></div>',
                    unsafe_allow_html=True)
page = st.sidebar.radio("Section", ["🏠  Overview", "👁️  Perception", "📡  Coordination", "🧠  Decisions",
                                    "🌫️  Robustness", "🎬  Mission replay"], label_visibility="collapsed")
st.sidebar.markdown("---")
st.sidebar.caption("Multi-Agent Depth-Aware Autonomous Decision System Using Infrared Vision\n\n"
                   "B.Tech Final Year Project · Amrita Vishwa Vidyapeetham, Amritapuri")

# ---------------- pages ----------------
if page.endswith("Overview"):
    hero("Multi-agent thermal search & rescue",
         "Night. Fog. A city intersection. Three thermal-sighted robots share what they see and reach stranded people together.",
         ["Webots", "Thermal physics model", "YOLOv8s", "BoT-SORT", "Depth U-Net", "Kalman fusion", "MAPPO"])
    cards([("🎯", "0.967", "Detection mAP50", "simulation test · +0.375 after fine-tuning", C["blue"]),
           ("📏", "1.89 m", "Depth RMSE", "monocular thermal U-Net", C["sky"]),
           ("🗺️", "0.08 m", "Shared-map error", "3 agents fused · 100% coverage", C["teal"]),
           ("🚑", "95%", "Mission success", "20 seeded Webots missions", C["indigo"]),
           ("🌫️", "90% vs 9%", "Thermal vs RGB", "person recall · night + fog", C["dteal"])])
    section("How it works")
    steps = [("Thermal cameras", "pole · robot · drone"), ("Detection", "YOLOv8s"), ("Tracking", "BoT-SORT"),
             ("Distance", "depth net + ground plane"), ("Map position", "with uncertainty"), ("Radio", "share beliefs"),
             ("Fusion", "Kalman shared map"), ("Decisions", "rules / MAPPO"), ("Act safely", "approach · avoid · yield")]
    html = '<div class="flow">' + '<span class="arrow">→</span>'.join(
        f'<div class="step"><div class="n">{i}</div><div><div class="t">{t}</div><div class="d">{d}</div></div></div>'
        for i, (t, d) in enumerate(steps, 1)) + "</div>"
    st.markdown(html, unsafe_allow_html=True)
    section("Deliverables")
    deliv = [("1 · Detection & tracking", "Fine-tuned thermal YOLOv8s, BoT-SORT tracking", True),
             ("2 · Depth estimation", "U-Net depth + ground-plane geometry", True),
             ("3 · Multi-agent coordination", "Radio sharing + Kalman belief fusion", True),
             ("4 · Decision engine", "Rule-based and MAPPO, compared in Webots", True),
             ("5 · Simulation", "Webots city world, 3 agents, fog, batch runs", True),
             ("6 · Evaluation & dashboard", "Ablations: comms, pole, fog, RGB", True),
             ("7 · Report & paper", "Writing phase", False)]
    for row in (deliv[:4], deliv[4:]):
        cols = st.columns(4)
        for col, (h, b, done) in zip(cols, row):
            chip = '<span class="chip">✓ Done</span>' if done else '<span class="chip wait">Next</span>'
            col.markdown(f'<div class="deliv"><div class="h">{h}{chip}</div><div class="b">{b}</div></div>',
                         unsafe_allow_html=True)
    section("Research questions")
    t1, t2, t3 = st.tabs(["RQ1 · Perception", "RQ2 · Cooperation", "RQ3 · Low visibility"])
    t1.markdown("**Can thermal perception detect, track and localise people reliably?**  \n"
                "Yes: 0.967 mAP50, 0.08 m shared-map error with three agents, depth RMSE 1.89 m.")
    t2.markdown("**Does sharing beliefs beat working alone, and at what communication cost?**  \n"
                "Fusing three viewpoints cuts map error 9× versus the best single agent. Mission success falls "
                "95% → 80% → 55% as message loss goes 0% → 50% → 100%.")
    t3.markdown("**How robust is the system at night and in fog?**  \n"
                "Missions stay at 90–100% success up to fog 0.09 and break at 0.15. At night in fog 0.09, thermal "
                "finds 90% of people where a normal camera finds 9%.")

elif page.endswith("Perception"):
    hero("Perception", "Seeing people in heat: detection, tracking and distance", ["YOLOv8s", "BoT-SORT", "U-Net · ResNet-34"])
    t1, t2, t3 = st.tabs(["🎯 Detection", "🔁 Tracking", "📏 Depth"])
    with t1:
        cards([("📈", "0.592 → 0.967", "mAP50 on simulation", "after fine-tuning", C["blue"]),
               ("🛡️", "0.740 → 0.740", "mAP50 on real FLIR", "no forgetting", C["teal"]),
               ("🧪", "2,400", "auto-labelled images", "800 per agent", C["indigo"])])
        fig = go.Figure([go.Bar(name="Before fine-tuning", x=["All", "Person", "Car"], y=[0.592, 0.814, 0.371], marker_color=C["slate"],
                                text=["0.59", "0.81", "0.37"], textposition="outside"),
                         go.Bar(name="After fine-tuning", x=["All", "Person", "Car"], y=[0.967, 0.981, 0.953], marker_color=C["blue"],
                                text=["0.97", "0.98", "0.95"], textposition="outside")])
        fig.update_layout(barmode="group", title="Simulation test set · mAP50", yaxis_range=[0, 1.12])
        st.plotly_chart(style(fig), use_container_width=True)
        with st.expander("Full detection results"):
            show_md("detection_results.md")
    with t2:
        c1, c2 = st.columns([1, 1])
        gif = f"{ROOT}/outputs/tracking/pole_cam_tracking.gif"
        if os.path.exists(gif):
            c1.image(gif, caption="Pole camera · detection with track IDs")
        fig = go.Figure([go.Bar(name="ByteTrack", x=["Recall", "MOTA"], y=[0.262, 0.227], marker_color=C["slate"],
                                text=["0.26", "0.23"], textposition="outside"),
                         go.Bar(name="BoT-SORT (camera-motion compensation)", x=["Recall", "MOTA"], y=[0.715, 0.633],
                                marker_color=C["teal"], text=["0.72", "0.63"], textposition="outside")])
        fig.update_layout(barmode="group", title="Drone (moving camera) · tracker comparison", yaxis_range=[0, 1])
        c2.plotly_chart(style(fig, 380), use_container_width=True)
        with st.expander("Full tracking results"):
            show_md("tracking_replay.md")
    with t3:
        t = md_table("depth_results.md")
        if t is not None and {"Method", "Agent", "RMSE (m)"} <= set(t.columns):
            t["RMSE (m)"] = pd.to_numeric(t["RMSE (m)"], errors="coerce")
            fig = go.Figure()
            for i, (meth, g) in enumerate(t.groupby("Method", sort=False)):
                fig.add_bar(name=meth, x=g["Agent"], y=g["RMSE (m)"], marker_color=[C["blue"], C["slate"]][i % 2],
                            text=[f"{v:.2f}" for v in g["RMSE (m)"]], textposition="outside")
            fig.update_layout(barmode="group", title="Depth error per agent (lower is better)", yaxis_title="RMSE (m)")
            st.plotly_chart(style(fig), use_container_width=True)
        with st.expander("Full depth results"):
            show_md("depth_results.md")

elif page.endswith("Coordination"):
    hero("Coordination", "From pixels to one shared map, over an imperfect radio", ["GPS + IMU", "Kalman filter", "Radio · message loss"])
    cards([("👤", "0.72 m", "Best single agent", "pole camera alone", C["slate"]),
           ("🤝", "0.08 m", "Three agents fused", "crossing lines of sight", C["teal"]),
           ("🆔", "4 / 4", "Identities kept", "0 ID switches on the map", C["blue"])])
    t1, t2, t3 = st.tabs(["📶 Message loss", "🧩 Fusion", "📍 Localisation"])
    with t1:
        df = summary([l for l in ("rule_v5", "rule_loss50", "rule_loss100") if l in batches()])
        if not df.empty:
            df["Loss"] = df["Setup"].map({"rule_v5": "0%", "rule_loss50": "50%", "rule_loss100": "100%"})
            fig = go.Figure()
            fig.add_bar(x=df["Loss"], y=df["Success %"], name="Success %", marker_color=[C["teal"], C["sky"], C["blue"]],
                        text=[f"{v}%" for v in df["Success %"]], textposition="outside")
            fig.add_scatter(x=df["Loss"], y=df["Robot-caused collisions"], name="Robot-caused collisions", yaxis="y2",
                            mode="lines+markers", line=dict(color=C["rose"], width=4), marker=dict(size=14))
            fig.update_layout(title="Mission success and safety vs message loss", xaxis_title="messages lost",
                              yaxis=dict(title="Success %", range=[0, 115]), yaxis2=dict(title="Collisions", overlaying="y", side="right"))
            st.plotly_chart(style(fig, 460), use_container_width=True)
            st.dataframe(df.drop(columns="Loss"), hide_index=True, use_container_width=True)
    with t2:
        show_md("fusion_results.md")
        show_md("live_fusion_results.md")
    with t3:
        show_md("localization_results.md")
        show_md("world_localization_results.md")

elif page.endswith("Decisions"):
    hero("Decisions", "Hand-written rules vs a trained multi-agent team", ["Task allocation", "Safe navigation", "MAPPO · CTDE"])
    all_b = batches()
    t1, t2, t3 = st.tabs(["📊 Compare batches", "⚖️ Rule vs RL", "🛠️ History"])
    with t1:
        default = [b for b in ("rule_v5", "rl_v1", "rl_v2", "rule_nopole_v3") if b in all_b]
        chosen = st.multiselect("Batches", all_b, default=default)
        if chosen:
            fig = go.Figure()
            for i, l in enumerate(chosen):
                b = load_batch(l)
                if b is not None:
                    col = SEQ[i % len(SEQ)]
                    fig.add_box(y=b["time"], name=l, boxpoints="all", jitter=0.45, pointpos=0, fillcolor=col + "22",
                                marker=dict(color=col, size=8, opacity=.85), line=dict(color=col),
                                hovertext=[f"seed {s}" for s in b["seed"]])
            fig.update_layout(title="Time to reach all people · every mission (failures at 180 s)", yaxis_title="seconds",
                              showlegend=False)
            st.plotly_chart(style(fig, 470), use_container_width=True)
            st.dataframe(summary(chosen), hide_index=True, use_container_width=True)
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
            cards([("⏱️", f"{d.mean():+.1f} s", f"Mean difference ({bb} − {a})", f"95% CI [{lo:+.1f}, {hi:+.1f}] s", C["blue"]),
                   ("⬇️", f"{(d < -0.5).sum()}", f"Missions {bb} was faster", "below the diagonal", C["green"]),
                   ("⬆️", f"{(d > 0.5).sum()}", f"Missions {bb} was slower", "above the diagonal", C["rose"])])
            lim = [70, 185]
            fig = go.Figure()
            fig.add_scatter(x=lim, y=lim, mode="lines", line=dict(color="#cbd5e1", dash="dash"), name="equal time")
            fig.add_scatter(x=m["time_a"], y=m["time_b"], mode="markers+text", text=m["seed"], textposition="top center",
                            marker=dict(size=15, color=d, colorscale=[[0, C["green"]], [0.5, "#e2e8f0"], [1, C["rose"]]],
                                        cmin=-15, cmax=15, showscale=True, colorbar=dict(title="Δ s"),
                                        line=dict(color="white", width=1.5)), name="mission (seed)")
            fig.update_layout(title=f"Each dot is one mission · below the line = {bb} faster", xaxis_title=f"{a} time (s)",
                              yaxis_title=f"{bb} time (s)", xaxis_range=lim, yaxis_range=lim)
            st.plotly_chart(style(fig, 540), use_container_width=True)
            st.caption("If the confidence interval contains 0, the difference is not statistically significant.")
    with t3:
        show_md("rule_based_results.md")

elif page.endswith("Robustness"):
    hero("Robustness", "What happens when the night gets darker and the fog gets thicker", ["Fog sweep", "Night", "RGB baseline"])
    t1, t2 = st.tabs(["🌫️ Fog", "🌙 Thermal vs RGB"])
    with t1:
        fogs = sorted([b for b in batches() if b.startswith("fog_")], key=lambda x: float(x[4:]))
        fig = go.Figure()
        fig.add_vrect(x0=0, x1=0.04, fillcolor=C["teal"], opacity=0.10, line_width=0, annotation_text="training range",
                      annotation_position="top left")
        df = None
        if fogs:
            df = summary(fogs)
            df["Fog"] = df["Setup"].str[4:].astype(float)
            fig.add_scatter(x=df["Fog"], y=df["Success %"] / 100, name="Mission success", mode="lines+markers",
                            line=dict(color=C["blue"], width=5, shape="spline"), marker=dict(size=14, line=dict(color="white", width=2)))
        t = md_table("fog_detection_curve.md")
        if t is not None:
            fig.add_scatter(x=pd.to_numeric(t.iloc[:, 0], errors="coerce"), y=pd.to_numeric(t.iloc[:, 2], errors="coerce"),
                            name="Detection recall", mode="lines+markers", line=dict(color=C["teal"], width=4, dash="dot", shape="spline"),
                            marker=dict(size=11))
        fig.update_layout(title="Detection and mission success vs fog", xaxis_title="fog attenuation per metre",
                          yaxis=dict(range=[0, 1.08], tickformat=".0%"))
        st.plotly_chart(style(fig, 480), use_container_width=True)
        if df is not None:
            st.dataframe(df.drop(columns="Fog"), hide_index=True, use_container_width=True)
    with t2:
        t = md_table("rgb_vs_thermal.md")
        if t is not None:
            light = st.radio("Lighting", ["☀️ Day", "🌙 Night"], horizontal=True, index=1)
            word = "Day" if "Day" in light else "Night"
            rows = t[t.iloc[:, 0].str.startswith(word)]
            fogx = rows.iloc[:, 0].str.extract(r"fog ([\d.]+)")[0]
            rgb = rows.iloc[:, 1].str.split("/").str[0].astype(float)
            th = rows.iloc[:, 2].str.split("/").str[0].astype(float)
            fig = go.Figure([go.Bar(name="Normal camera (RGB, COCO YOLOv8s)", x=fogx, y=rgb, marker_color=C["slate"],
                                    text=[f"{v:.0%}" for v in rgb], textposition="outside"),
                             go.Bar(name="Thermal camera (ours)", x=fogx, y=th, marker_color=C["blue"],
                                    text=[f"{v:.0%}" for v in th], textposition="outside")])
            fig.update_layout(barmode="group", title=f"Person detection recall · {word.lower()}",
                              xaxis_title="fog attenuation per metre", yaxis=dict(range=[0, 1.12], tickformat=".0%"))
            st.plotly_chart(style(fig, 460), use_container_width=True)
        img = f"{RES}/rgb_vs_thermal_example.png"
        if os.path.exists(img):
            st.image(img, caption="Same moment · RGB by day · RGB at night in fog · thermal in fog (blue = true people, green = detected)")
        with st.expander("Full table and method"):
            show_md("rgb_vs_thermal.md")

else:
    hero("Mission replay", "Press ▶ Play to watch a recorded mission on the map", ["True positions", "Shared map", "Rescues"])
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
    cards([("🧠", str(m.get("policy")), "Policy", f"seed {m.get('seed')}", C["blue"]),
           ("🌫️", str((m.get("conditions") or {}).get("fog")), "Fog", "attenuation per metre", C["sky"]),
           ("⏱️", f"{m['end_t']:.0f} s", "Mission end", m["reason"], C["indigo"]),
           ("⭐", str(len(m["rescues"])), "Rescues claimed", f"pole camera: {m.get('pole', True)}", C["amber"])])
    st.write("")
    tt = np.array([r["t"] for r in truth])
    ft = np.array([f["t"] for f in frames])
    times = np.arange(1.0, float(m["end_t"]) + 0.01, 1.0)
    xy = lambda L: ([p[0] for p in L], [p[1] for p in L])

    def frame_data(t):
        r = truth[int(np.argmin(np.abs(tt - t)))]
        people = [o for o in r["objects"] if o["cls"] == 0]
        stand = [o["pos"][:2] for o in people if o["name"] not in ("pedestrian(1)", "pedestrian(2)")]
        walk = [o["pos"][:2] for o in people if o["name"] in ("pedestrian(1)", "pedestrian(2)")]
        trail = [next(o["pos"][:2] for o in q["objects"] if o["name"] == "ugv") for q in truth if q["t"] <= t]
        fused = [q["xy"] for q in frames[int(np.argmin(np.abs(ft - t)))].get("fused", [])]
        resc = [c["xy"] for c in m["rescues"] if c["t"] <= t]
        return [
            go.Scatter(x=xy(trail)[0], y=xy(trail)[1], mode="lines", line=dict(color=C["sky"], width=4), name="robot trail"),
            go.Scatter(x=xy(fused)[0], y=xy(fused)[1], mode="markers", name="robot's shared map",
                       marker=dict(size=30, color="rgba(20,184,166,0.15)", line=dict(color=C["teal"], width=3))),
            go.Scatter(x=xy(stand)[0], y=xy(stand)[1], mode="markers", name="person (standing)",
                       marker=dict(size=15, color=C["blue"], line=dict(color="white", width=2))),
            go.Scatter(x=xy(walk)[0], y=xy(walk)[1], mode="markers", name="person (walking)",
                       marker=dict(size=17, color=C["indigo"], symbol="triangle-up", line=dict(color="white", width=2))),
            go.Scatter(x=[trail[-1][0]] if trail else [], y=[trail[-1][1]] if trail else [], mode="markers", name="ground robot",
                       marker=dict(size=20, color=C["navy"], symbol="square", line=dict(color=C["sky"], width=3))),
            go.Scatter(x=xy(resc)[0], y=xy(resc)[1], mode="markers", name="rescue claimed",
                       marker=dict(size=28, color=C["amber"], symbol="star", line=dict(color="white", width=1.5))),
        ]

    fig = go.Figure(data=frame_data(times[0]), frames=[go.Frame(data=frame_data(t), name=f"{t:.0f}") for t in times])
    # approximate road layout around the intersection (decoration)
    for shp in (dict(x0=39, x1=51, y0=-62, y1=-8), dict(x0=28, x1=62, y0=-51, y1=-39)):
        fig.add_shape(type="rect", fillcolor="#e2e8f0", line=dict(width=0), layer="below", **shp)
    fig.add_shape(type="line", x0=45, x1=45, y0=-62, y1=-8, line=dict(color="#ffffff", width=2, dash="dash"), layer="below")
    fig.add_shape(type="line", x0=28, x1=62, y0=-45, y1=-45, line=dict(color="#ffffff", width=2, dash="dash"), layer="below")
    fig.add_shape(type="rect", x0=36, y0=-54, x1=54, y1=-36, line=dict(color="#94a3b8", dash="dot"), layer="below")
    fig.add_annotation(x=29.5, y=-28, text="📹 pole camera", showarrow=False, font=dict(size=12, color="#334155"), xanchor="left")
    fig.update_layout(
        xaxis=dict(range=[28, 62], title="east (m)", showgrid=False), yaxis=dict(range=[-60, -12], title="north (m)",
        scaleanchor="x", showgrid=False),
        updatemenus=[dict(type="buttons", x=0, y=-0.09, xanchor="left", direction="left", bgcolor="#ffffff",
                          bordercolor="#cbd5e1", font=dict(color="#0f172a"), buttons=[
            dict(label="▶ Play", method="animate", args=[None, dict(frame=dict(duration=220, redraw=True), fromcurrent=True)]),
            dict(label="⏸ Pause", method="animate", args=[[None], dict(mode="immediate", frame=dict(duration=0))])])],
        sliders=[dict(active=0, y=-0.03, currentvalue=dict(prefix="t = ", suffix=" s", font=dict(color="#0f172a")),
                      steps=[dict(method="animate", label=f"{t:.0f}", args=[[f"{t:.0f}"], dict(mode="immediate",
                              frame=dict(duration=0, redraw=True))]) for t in times])])
    st.plotly_chart(style(fig, 740), use_container_width=True)
    section("Rescue timeline")
    tl = go.Figure()
    tl.add_scatter(x=[0, m["end_t"]], y=[0, 0], mode="lines", line=dict(color="#cbd5e1", width=6), showlegend=False)
    tl.add_scatter(x=[c["t"] for c in m["rescues"]], y=[0] * len(m["rescues"]), mode="markers+text", showlegend=False,
                   text=[f"#{i} · {c['t']:.0f}s" for i, c in enumerate(m["rescues"], 1)], textposition="top center",
                   marker=dict(size=22, color=C["amber"], symbol="star", line=dict(color="white", width=1.5)))
    tl.update_layout(yaxis=dict(visible=False, range=[-1, 1.5]), xaxis=dict(title="mission time (s)", range=[0, max(180, m["end_t"])]))
    st.plotly_chart(style(tl, 200), use_container_width=True)
    st.caption("Grey bands: roads (approximate). Dotted square: where standing people can be placed. "
               "Teal rings: where the robot's shared map believes people are.")
