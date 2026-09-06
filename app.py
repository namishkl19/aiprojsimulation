"""
AI-Based Peer-to-Peer (P2P) and Vehicle-to-Grid (V2G) Energy Management System
with Non-Compliance Modeling and Adaptive Rescheduling
Interactive Streamlit Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pulp import LpProblem, LpVariable, LpMinimize, lpSum, LpStatus, PULP_CBC_CMD

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="AI P2P & V2G Energy Management",
    page_icon="\u26a1",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CONSTANTS
# ============================================================
HOURS = list(range(24))
DT = 1.0

def _tou_price(h):
    # Peak Hours
    if 6 <= h <= 9 or 17 <= h <= 21:
        return 11.0

    # Off-Peak Hours
    if h >= 22 or h <= 5:
        return 6.0

    # Standard Hours
    return 8.0

TOU = {h: _tou_price(h) for h in HOURS}
TOU_LABEL = {}
for h in HOURS:
    if 6 <= h <= 9 or 17 <= h <= 21:
        TOU_LABEL[h] = "Peak"

    elif h >= 22 or h <= 5:
        TOU_LABEL[h] = "Off-Peak"

    else:
        TOU_LABEL[h] = "Standard"

BAT_CAP   = 60
MAX_CH    = 11
MAX_DIS   = 10
MAX_P2P   = 7
MIN_SOC   = 0.10 * BAT_CAP
AUX_KW    = 0.3
V2G_FRAC  = 0.80
P2P_FRAC  = 0.90

C1  = "#1f77b4"
C2  = "#ff7f0e"
CGR = "#2ecc71"
CV2G= "#e74c3c"
CP2P= "#9b59b6"

# ============================================================
# SOLVERS
# ============================================================
def _milp(s1, s2, tgt, a1, d1, a2, d2):
    prob = LpProblem("P2P_V2G", LpMinimize)
    E = [1, 2]
    arr = {1: a1, 2: a2}
    dep = {1: min(d1, 24), 2: min(d2, 24)}
    s0  = {1: np.clip(s1, 10, 100) * BAT_CAP / 100,
           2: np.clip(s2, 10, 100) * BAT_CAP / 100}
    sT  = {1: np.clip(tgt, 50, 100) * BAT_CAP / 100,
           2: np.clip(tgt, 50, 100) * BAT_CAP / 100}

    ch  = {(i,t): LpVariable(f"c{i}_{t}", 0, MAX_CH)  for i in E for t in HOURS}
    dis = {(i,t): LpVariable(f"d{i}_{t}", 0, MAX_DIS) for i in E for t in HOURS}
    pp  = {(i,t): LpVariable(f"p{i}_{t}", 0, MAX_P2P) for i in E for t in HOURS}
    so  = {(i,t): LpVariable(f"s{i}_{t}", 0, BAT_CAP) for i in E for t in range(25)}
    gi  = {t: LpVariable(f"gi{t}", 0, 22) for t in HOURS}
    ge  = {t: LpVariable(f"ge{t}", 0, 10) for t in HOURS}

    prob += (lpSum(TOU[t]*gi[t]*DT for t in HOURS)
             - lpSum(V2G_FRAC*TOU[t]*ge[t]*DT for t in HOURS))

    for t in HOURS:
        prob += gi[t] == lpSum(ch[i,t] for i in E)
        prob += ge[t] == lpSum(dis[i,t] for i in E)

    for i in E:
        o = 3 - i
        prob += so[i, arr[i]] == s0[i]
        for t in range(arr[i], dep[i]):
            prob += so[i,t+1] == so[i,t] + (
                ch[i,t] - dis[i,t] + pp[o,t] - pp[i,t])*DT - AUX_KW*DT
            prob += so[i,t] >= MIN_SOC
        prob += so[i, dep[i]] >= sT[i]
        for t in HOURS:
            if t < arr[i] or t >= dep[i]:
                prob += ch[i,t] == 0
                prob += dis[i,t] == 0
                prob += pp[i,t] == 0

    status = prob.solve(PULP_CBC_CMD(msg=0, timeLimit=30))
    if LpStatus[status] != "Optimal":
        return None

    r = dict(
        ch  = {(i,t): round(ch[i,t].varValue or 0, 4)  for i in E for t in HOURS},
        dis = {(i,t): round(dis[i,t].varValue or 0, 4) for i in E for t in HOURS},
        pp  = {(i,t): round(pp[i,t].varValue or 0, 4)  for i in E for t in HOURS},
        soc = {(i,t): round(so[i,t].varValue or 0, 4)  for i in E for t in range(25)},
        gi  = {t: round(gi[t].varValue or 0, 4) for t in HOURS},
        ge  = {t: round(ge[t].varValue or 0, 4) for t in HOURS},
    )
    r["gcost"]  = sum(TOU[t]*r["gi"][t] for t in HOURS)
    r["vrevenue"]= sum(V2G_FRAC*TOU[t]*r["ge"][t] for t in HOURS)
    r["net"]    = r["gcost"] - r["vrevenue"]
    r["gkwh"]   = sum(r["gi"][t] for t in HOURS)
    r["vkwh"]   = sum(r["ge"][t] for t in HOURS)
    r["pkwh"]   = sum(r["pp"][i,t] for i in E for t in HOURS)
    r["fsoc"]   = {i: r["soc"][i, dep[i]] for i in E}
    r["asoc"]   = np.mean([r["fsoc"][i]/BAT_CAP*100 for i in E])
    r["dep"]    = dep
    return r


def _baseline(s1, s2, tgt, a1, d1, a2, d2):
    prob = LpProblem("BL", LpMinimize)
    E = [1, 2]
    arr = {1: a1, 2: a2}
    dep = {1: min(d1,24), 2: min(d2,24)}
    s0  = {1: np.clip(s1,10,100)*BAT_CAP/100, 2: np.clip(s2,10,100)*BAT_CAP/100}
    sT  = {1: np.clip(tgt,50,100)*BAT_CAP/100, 2: np.clip(tgt,50,100)*BAT_CAP/100}

    ch = {(i,t): LpVariable(f"bc{i}_{t}", 0, MAX_CH) for i in E for t in HOURS}
    so = {(i,t): LpVariable(f"bs{i}_{t}", 0, BAT_CAP) for i in E for t in range(25)}

    prob += lpSum(TOU[t]*ch[i,t]*DT for i in E for t in HOURS)

    for i in E:
        prob += so[i, arr[i]] == s0[i]
        for t in range(arr[i], dep[i]):
            prob += so[i,t+1] == so[i,t] + ch[i,t]*DT - AUX_KW*DT
            prob += so[i,t] >= MIN_SOC
        prob += so[i, dep[i]] >= sT[i]
        for t in HOURS:
            if t < arr[i] or t >= dep[i]:
                prob += ch[i,t] == 0

    status = prob.solve(PULP_CBC_CMD(msg=0, timeLimit=30))
    if LpStatus[status] != "Optimal":
        return None

    r = dict(
        ch  = {(i,t): round(ch[i,t].varValue or 0,4)  for i in E for t in HOURS},
        dis = {(i,t): 0 for i in E for t in HOURS},
        pp  = {(i,t): 0 for i in E for t in HOURS},
        soc = {(i,t): round(so[i,t].varValue or 0,4)  for i in E for t in range(25)},
        gi  = {t: sum(round(ch[i,t].varValue or 0,4) for i in E) for t in HOURS},
        ge  = {t: 0 for t in HOURS},
    )
    r["gcost"]   = sum(TOU[t]*r["gi"][t] for t in HOURS)
    r["vrevenue"]= 0
    r["net"]     = r["gcost"]
    r["gkwh"]    = sum(r["gi"][t] for t in HOURS)
    r["vkwh"]    = 0
    r["pkwh"]    = 0
    r["fsoc"]    = {i: r["soc"][i, dep[i]] for i in E}
    r["asoc"]    = np.mean([r["fsoc"][i]/BAT_CAP*100 for i in E])
    r["dep"]     = dep
    return r

# ============================================================
# SIMULATION RUNNER
# ============================================================
def run_sim(s1, s2, tgt, scenario):
    a1, d1 = 8, 18
    a2, d2 = 10, 20
    oa2, od1, os1 = a2, d1, s1

    aa2, ad1, as1 = a2, d1, s1
    if scenario == "late_arrival":
        aa2 = a2 + 2
    elif scenario == "early_departure":
        ad1 = d1 - 2
    elif scenario == "battery_mismatch":
        as1 = max(10, s1 - 10)

    base = _baseline(s1, s2, tgt, a1, d1, a2, d2)
    opt0 = _milp(s1, s2, tgt, a1, d1, a2, d2)
    if base is None or opt0 is None:
        return None

    adapted = None
    if scenario != "normal":
        adapted = _milp(as1, s2, tgt, a1, ad1, aa2, d2)

    events = []
    if scenario == "late_arrival":
        events.append(dict(Time=f"{a2:02d}:00", Type="\u26a0\ufe0f Late Plug-in",
            Detail=f"EV2 arrived at {aa2:02d}:00 instead of {oa2:02d}:00",
            kW=0, Impact="System rescheduled"))
    elif scenario == "early_departure":
        events.append(dict(Time=f"{ad1:02d}:00", Type="\u26a0\ufe0f Early Unplug",
            Detail=f"EV1 left at {ad1:02d}:00 instead of {od1:02d}:00",
            kW=0, Impact="System rescheduled"))
    elif scenario == "battery_mismatch":
        events.append(dict(Time=f"{a1:02d}:00", Type="\u26a0\ufe0f Battery Mismatch",
            Detail=f"EV1 actual SoC {as1:.0f}% (expected {os1:.0f}%)",
            kW=0, Impact="System rescheduled"))

    active = adapted if adapted else opt0
    for t in HOURS:
        if active["gi"][t] > 0.01:
            events.append(dict(Time=f"{t:02d}:00", Type="\u26a1 Grid Charge",
                Detail=f"Importing at {active['gi'][t]:.1f} kW",
                kW=round(active["gi"][t],2),
                Impact=f"+₹{TOU[t]*active['gi'][t]:.2f}"))
        if active["ge"][t] > 0.01:
            events.append(dict(Time=f"{t:02d}:00", Type="\U0001f50b V2G Sale",
                Detail=f"Exporting at {active['ge'][t]:.1f} kW",
                kW=round(active["ge"][t],2),
                Impact=f"-₹{V2G_FRAC*TOU[t]*active['ge'][t]:.2f}"))
        if active["pp"][1,t] > 0.01:
            events.append(dict(Time=f"{t:02d}:00", Type="\U0001f504 P2P EV1\u2192EV2",
                Detail=f"Transferring at {active['pp'][1,t]:.1f} kW",
                kW=round(active["pp"][1,t],2), Impact="System saving"))
        if active["pp"][2,t] > 0.01:
            events.append(dict(Time=f"{t:02d}:00", Type="\U0001f504 P2P EV2\u2192EV1",
                Detail=f"Transferring at {active['pp'][2,t]:.1f} kW",
                kW=round(active["pp"][2,t],2), Impact="System saving"))
    events.sort(key=lambda x: x["Time"])

    return dict(base=base, opt0=opt0, adapted=adapted, events=events, scenario=scenario)

# ============================================================
# EXPLANATION GENERATOR
# ============================================================
def explain(sim):
    base, opt0 = sim["base"], sim["opt0"]
    adapted = sim["adapted"]
    sc = sim["scenario"]
    active = adapted if adapted else opt0

    sav = base["net"] - active["net"]
    pct = sav / base["net"] * 100 if base["net"] > 0 else 0

    lines = []
    lines.append(
        f"**\U0001f4b0 Cost Savings:** The AI saved **₹{sav:.2f}** ({pct:.1f}%) "
        f"compared to standard grid-only charging (₹{base['net']:.2f} \u2192 ₹{active['net']:.2f})."
    )

    if active["pkwh"] > 0.1:
        lines.append(
            f"**\U0001f504 P2P Trading:** **{active['pkwh']:.1f} kWh** was traded between EVs "
            f"internally, avoiding expensive grid purchases during peak hours."
        )

    if active["vkwh"] > 0.1:
        lines.append(
            f"**\U0001f50b V2G Revenue:** **{active['vkwh']:.1f} kWh** was sold back to the "
            f"grid, earning **₹{active['vrevenue']:.2f}**."
        )

    peak_gi = sum(active["gi"][t] for t in HOURS if TOU_LABEL[t] == "Peak")
    off_gi  = sum(active["gi"][t] for t in HOURS if TOU_LABEL[t] == "Off-Peak")
    if peak_gi < off_gi and (peak_gi + off_gi) > 0:
        lines.append(
            f"**\U0001f319 Smart Scheduling:** The AI shifted most charging to off-peak "
            f"hours (₹6/kWh) and avoided peak hours (₹11/kWh)."
        )

    if sc == "late_arrival":
        lines.append(
            "**\u26a0\ufe0f Non-Compliance (Late Arrival):** EV2 arrived 2 hours late. "
            "The AI rescheduled P2P transfers and increased off-peak charging to compensate."
        )
    elif sc == "early_departure":
        lines.append(
            "**\u26a0\ufe0f Non-Compliance (Early Unplug):** EV1 left 2 hours early. "
            "The AI accelerated charging and used P2P to meet the target."
        )
    elif sc == "battery_mismatch":
        lines.append(
            "**\u26a0\ufe0f Non-Compliance (Battery Mismatch):** EV1 had 10% less charge "
            "than expected. The AI increased grid charging during off-peak hours."
        )

    return lines

# ============================================================
# CHART BUILDERS
# ============================================================
def _hour_color(h):
    p = TOU[h]
    if p == 0.30:
        return "rgba(231,76,60,0.10)"
    if p == 0.10:
        return "rgba(46,204,113,0.10)"
    return "rgba(243,156,18,0.10)"

def build_soc_chart(opt0, base, adapted, scenario):
    fig = go.Figure()

    hours_list = list(range(25))

    def add_trace(soc_dict, ev, name, color, dash):
        y = [soc_dict[(ev, t)] / BAT_CAP * 100 for t in hours_list]
        fig.add_trace(go.Scatter(
            x=hours_list, y=y, mode="lines+markers", name=name,
            line=dict(color=color, width=3, dash=dash),
            marker=dict(size=4), visible=True,
        ))

    add_trace(base["soc"],  1, "EV1 \u2014 Baseline", C1, "dot")
    add_trace(base["soc"],  2, "EV2 \u2014 Baseline", C2, "dot")
    add_trace(opt0["soc"],  1, "EV1 \u2014 AI Plan",  C1, "solid")
    add_trace(opt0["soc"],  2, "EV2 \u2014 AI Plan",  C2, "solid")

    if adapted:
        add_trace(adapted["soc"], 1, "EV1 \u2014 Adapted", C1, "dashdot")
        add_trace(adapted["soc"], 2, "EV2 \u2014 Adapted", C2, "dashdot")

    for h in HOURS:
        fig.add_vrect(x0=h-0.5, x1=h+0.5, fillcolor=_hour_color(h),
                       layer="below", line_width=0)

    fig.add_hline(y=80, line=dict(color="gray", dash="dash", width=1),
                  annotation_text="Target 80%", annotation_position="top right")

    fig.update_layout(
        title="Battery State of Charge \u2014 24-Hour Tracker",
        xaxis_title="Hour of Day", yaxis_title="State of Charge (%)",
        yaxis=dict(range=[0, 105]), xaxis=dict(dtick=1),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        height=420, margin=dict(t=80, b=40),
        template="plotly_white",
    )
    return fig


def build_power_chart(active):
    gi = [active["gi"][t] for t in HOURS]
    ge = [-active["ge"][t] for t in HOURS]
    pp1 = [active["pp"][1,t] for t in HOURS]
    pp2 = [active["pp"][2,t] for t in HOURS]

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=HOURS, y=gi, name="Grid \u2192 EV (Import)",
                          marker_color=CGR, opacity=0.85))
    fig.add_trace(go.Bar(x=HOURS, y=ge, name="EV \u2192 Grid (V2G)",
                          marker_color=CV2G, opacity=0.85))
    fig.add_trace(go.Scatter(x=HOURS, y=pp1, name="P2P EV1\u2192EV2",
                              mode="lines+markers", line=dict(color=CP2P, width=2),
                              marker=dict(size=6)), secondary_y=True)
    fig.add_trace(go.Scatter(x=HOURS, y=pp2, name="P2P EV2\u2192EV1",
                              mode="lines+markers", line=dict(color="#e67e22", width=2),
                              marker=dict(size=6, symbol="diamond")), secondary_y=True)

    fig.update_layout(
        barmode="relative",
        title="Power Flow \u2014 Grid, V2G & P2P Energy Direction",
        xaxis_title="Hour of Day", xaxis=dict(dtick=1),
        yaxis_title="Grid Power (kW)",
        yaxis2_title="P2P Power (kW)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        height=400, margin=dict(t=80, b=40),
        template="plotly_white",
    )

    for h in HOURS:
        fig.add_vrect(x0=h-0.5, x1=h+0.5, fillcolor=_hour_color(h),
                       layer="below", line_width=0, row=1, col=1)

    return fig


def build_price_chart(active):
    prices = [TOU[t] for t in HOURS]

    actions = []
    for t in HOURS:
        gi = active["gi"][t]
        ge = active["ge"][t]
        pp = active["pp"][1,t] + active["pp"][2,t]
        if gi > 0.05 and ge < 0.05 and pp < 0.05:
            actions.append("Charge")
        elif ge > 0.05 and gi < 0.05:
            actions.append("V2G Sell")
        elif pp > 0.05:
            actions.append("P2P Trade")
        else:
            actions.append("Hold")

    color_map = {
        "Charge":    "#2ecc71",
        "V2G Sell":  "#e74c3c",
        "P2P Trade": "#9b59b6",
        "Hold":      "#95a5a6",
    }
    symbol_map = {
        "Charge":    "arrow-up",
        "V2G Sell":  "arrow-down",
        "P2P Trade": "diamond",
        "Hold":      "square",
    }

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=HOURS, y=prices, mode="lines+markers", name="ToU Price (₹/kWh)",
        line=dict(color="#34495e", width=2),
        marker=dict(size=6),
    ))

    for act in set(actions):
        idx = [i for i, a in enumerate(actions) if a == act]
        fig.add_trace(go.Scatter(
            x=[HOURS[i] for i in idx],
            y=[prices[i] for i in idx],
            mode="markers", name=f"AI Decision: {act}",
            marker=dict(color=color_map[act], size=14, symbol=symbol_map[act],
                        line=dict(width=1, color="white")),
        ))

    for h in HOURS:
        fig.add_vrect(x0=h-0.5, x1=h+0.5, fillcolor=_hour_color(h),
                       layer="below", line_width=0)

    fig.update_layout(
        title="ToU Price vs AI Decision Overlay",
        xaxis_title="Hour of Day", xaxis=dict(dtick=1),
        yaxis_title="Price (₹/kWh)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        height=400, margin=dict(t=80, b=40),
        template="plotly_white",
    )

    annotations = []
    legend_text = {
        11.0: "Peak ₹11",
        8.0: "Standard ₹8",
        6.0: "Off-Peak ₹6",
    }
    seen = set()
    for h in HOURS:
        p = TOU[h]
        if p not in seen:
            annotations.append(dict(
                x=h, y=p + 0.02, text=legend_text[p],
                showarrow=False, font=dict(size=9, color="#555"),
            ))
            seen.add(p)

    fig.update_layout(annotations=annotations)
    return fig

# ============================================================
# STREAMLIT UI
# ============================================================

def main():
    st.markdown("""
    <div style="background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);
         padding:18px 24px;border-radius:12px;color:white;margin-bottom:18px;">
      <h2 style="margin:0;color:white;">AI-Based P2P &amp; V2G Energy Management System</h2>
      <p style="margin:6px 0 0 0;font-size:1.05em;opacity:0.95;">
        An AI system that lets Electric Vehicles trade power with each other (P2P) or sell
        back to the power grid (V2G) to lower charging costs and prevent grid overload.
      </p>
    </div>
    """, unsafe_allow_html=True)

    # --- Sidebar Controls ---
    with st.sidebar:
        st.header("\U0001f3ae Simulation Controls")
        soc1 = st.slider("Initial EV1 SoC (%)", 10, 90, 30, step=5)
        soc2 = st.slider("Initial EV2 SoC (%)", 10, 90, 40, step=5)
        tgt  = st.slider("Target SoC (%)", 50, 100, 80, step=5)
        scenario = st.selectbox("Non-Compliance Scenario", [
            ("normal",          "Option A: Normal Operation (Compliant)"),
            ("late_arrival",    "Option B: EV2 Arrives 2 Hours Late"),
            ("early_departure", "Option C: EV1 Leaves Early (Early Unplug)"),
            ("battery_mismatch","Option D: Battery Charge Mismatch"),
        ], format_func=lambda x: x[1])
        st.markdown("---")
        run_btn = st.button("\U0001f680 Run Live Adaptive Simulation", use_container_width=True)

    # --- Run or load cached ---
    if run_btn:
        with st.spinner("Running PuLP MILP optimization\u2026"):
            sim = run_sim(soc1, soc2, tgt, scenario[0])
        if sim is None:
            st.error("Solver could not find an optimal solution. Try adjusting parameters.")
            st.stop()
        st.session_state["sim"] = sim

    if "sim" not in st.session_state:
        st.info("Adjust the sidebar controls and click **Run** to start the simulation.")
        st.stop()

    sim = st.session_state["sim"]
    base, opt0 = sim["base"], sim["opt0"]
    adapted = sim["adapted"]
    active  = adapted if adapted else opt0
    sc      = sim["scenario"]

    # --- KPI Tiles ---
    sav = base["net"] - active["net"]
    pct = sav / base["net"] * 100 if base["net"] > 0 else 0
    warnings = 1 if sc != "normal" else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("\U0001f4b0 Total Cost Saved", f"{pct:.1f}%", f"₹{sav:.2f}")
    c2.metric("\u26a1 P2P Energy Shared", f"{active['pkwh']:.1f} kWh")
    c3.metric("\U0001f50b Avg Battery SoC", f"{active['asoc']:.1f}%")
    c4.metric("\u26a0\ufe0f Active Warnings", f"{warnings}",
              delta="Non-compliance detected" if warnings else "All clear",
              delta_color="inverse" if warnings else "off")

    st.markdown("---")

    # --- Tabs ---
    tab_soc, tab_pwr, tab_price = st.tabs([
        "\U0001f50b SoC Tracker",
        "\u26a1 Power Flow & Energy Direction",
        "\U0001f4b0 Price vs AI Decision",
    ])

    with tab_soc:
        st.plotly_chart(build_soc_chart(opt0, base, adapted, sc),
                        use_container_width=True)

    with tab_pwr:
        st.plotly_chart(build_power_chart(active), use_container_width=True)

    with tab_price:
        st.plotly_chart(build_price_chart(active), use_container_width=True)

    # --- Explanation Panel ---
    st.markdown("---")
    st.subheader("\U0001f4a1 What Just Happened?")
    for line in explain(sim):
        st.markdown(f"- {line}")

    # --- Transaction Log ---
    st.markdown("---")
    st.subheader("\U0001f4cb Transaction Log")
    if sim["events"]:
        df = pd.DataFrame(sim["events"])
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No transactions during this period.")

    # --- Footer ---
    st.markdown("---")
    st.caption(
        "Backend: PuLP MILP Solver | Frontend: Streamlit + Plotly | "
        "24-Hour Time Horizon | Time-of-Use Tariffs (Peak ₹11/kWh, Standard ₹8/kWh, Off-Peak ₹6/kWh)"
    )


if __name__ == "__main__":
    main()
