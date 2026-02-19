
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.integrate import solve_ivp

st.set_page_config(page_title="NCATS DMPK Core | PBPK Simulator", layout="wide")

CUSTOM_CSS = """
<style>
.header {
  background: linear-gradient(90deg, rgba(11,95,255,0.10), rgba(11,95,255,0.02), rgba(11,95,255,0.10));
  background-size: 200% 200%;
  animation: gradient 10s ease infinite;
  border: 1px solid rgba(17,24,39,0.08);
  border-radius: 18px;
  padding: 18px 18px 14px 18px;
}
@keyframes gradient {
  0% {background-position: 0% 50%;}
  50% {background-position: 100% 50%;}
  100% {background-position: 0% 50%;}
}
.badge {
  display:inline-block;
  padding: 4px 10px;
  border-radius: 999px;
  border: 1px solid rgba(11,95,255,0.25);
  color: rgba(11,95,255,1);
  background: rgba(11,95,255,0.06);
  font-size: 12px;
  margin-left: 10px;
}
.smallnote {
  color: rgba(17,24,39,0.68);
  font-size: 12px;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

with st.container():
    c1, c2 = st.columns([0.18, 0.82], vertical_alignment="center")
    with c1:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/6/6e/National_Center_for_Advancing_Translational_Sciences_logo.png/250px-National_Center_for_Advancing_Translational_Sciences_logo.png", caption=None, use_container_width=True)
    with c2:
        st.markdown(
            '<div class="header">'
            '<div style="display:flex;align-items:center;gap:10px;">'
            '<h2 style="margin:0;">PBPK Dose Simulator</h2>'
            '<span class="badge">NCATS DMPK Core</span>'
            '</div>'
            '<div class="smallnote">Interactive PBPK simulation with loading + maintenance regimens, tissue displays, and basic PK indicators (Cmax, AUC, Cavg).</div>'
            '</div>',
            unsafe_allow_html=True
        )

st.caption("DISCLAIMER: Research/education tool only. Not validated for clinical decision-making.")

@dataclass
class Params:
    VmaxL_livC: float = 0.760
    Km_liv: float = 0.0036
    Ka: float = 0.297
    Fab: float = 1.0
    Fup: float = 0.008
    Rbp: float = 1.5

    PL: float = 53.0
    PK: float = 57.8
    PS: float = 31.7
    PLu: float = 43.2
    PM: float = 5.9
    PP: float = 40.2
    PHe: float = 5.9
    PBR: float = 1.33
    Prest: float = 0.139
    PT: float = 29.9
    PInt: float = 0.139

    PABRC: float = 0.0012
    PATC: float = 0.283

    BW: float = 70.0
    VLuC: float = 0.016
    VBloodAC: float = 0.022
    VBloodVC: float = 0.044
    VMC: float = 0.395
    VHeC: float = 0.004
    VBRC: float = 0.023
    VKC: float = 0.004
    VLC: float = 0.019
    VITC: float = 0.032
    VSC: float = 0.002
    VTC: float = 0.0007
    VPC: float = 0.0014

    GFRC: float = 0.007

    QCC: float = 5.42634768
    QMC: float = 0.139829804
    QHeC: float = 0.041316184
    QBRC: float = 0.153309449
    QKC: float = 0.193121016
    QLC: float = 0.227367562
    QITC: float = 0.129996347
    QSC: float = 0.03051901
    QTC: float = 0.0158
    QPC: float = 0.01

    BVBR: float = 0.08
    BVT: float = 0.07

STATE = ["ALu","AM","AHe","AK","ABRb","ABRt","Arest","AS","AP","AGutLu","AI","ATb","ATt","AL","AV"]

def build_derived(p: Params) -> Dict[str, float]:
    BW = p.BW
    VL  = BW * p.VLC
    VK  = BW * p.VKC
    VS  = BW * p.VSC
    VM  = BW * p.VMC
    VP  = BW * p.VPC
    VIT = BW * p.VITC
    VHe = BW * p.VHeC
    VLu = BW * p.VLuC
    VBloodA = BW * p.VBloodAC
    VBloodV = BW * p.VBloodVC

    VBR  = BW * p.VBRC
    VBRb = VBR * p.BVBR
    VBRt = VBR - VBRb

    VT   = BW * p.VTC
    VTb  = VT * p.BVT
    VTt  = VT - VTb

    QC  = p.QCC * BW
    QL  = QC * p.QLC
    QBR = QC * p.QBRC
    QK  = QC * p.QKC
    QS  = QC * p.QSC
    QM  = QC * p.QMC
    QT  = QC * p.QTC
    QP  = QC * p.QPC
    QIT = QC * p.QITC
    QHe = QC * p.QHeC
    Qrest = QC - QL - QBR - QK - QM - QHe - QT

    GFR = p.GFRC * BW
    PABR = p.PABRC * QBR
    PAT  = p.PATC  * QT

    known_frac = (p.VLC + p.VKC + p.VSC + p.VMC + p.VPC + p.VITC + p.VHeC + p.VLuC + p.VBRC + p.VTC + p.VBloodAC + p.VBloodVC)
    Vrest = max(1e-6, BW * (1.0 - known_frac))

    return dict(
        VL=VL, VK=VK, VS=VS, VM=VM, VP=VP, VIT=VIT, VHe=VHe, VLu=VLu, VBloodA=VBloodA, VBloodV=VBloodV,
        VBR=VBR, VBRb=VBRb, VBRt=VBRt, VT=VT, VTb=VTb, VTt=VTt,
        QC=QC, QL=QL, QBR=QBR, QK=QK, QS=QS, QM=QM, QT=QT, QP=QP, QIT=QIT, QHe=QHe, Qrest=Qrest,
        GFR=GFR, PABR=PABR, PAT=PAT, Vrest=Vrest
    )

def rhs(t: float, y: np.ndarray, p: Params, d: Dict[str, float]) -> np.ndarray:
    VLu, VM, VHe, VK, VBRb, VBRt, Vrest, VS, VP, VIT, VTb, VTt, VL, VBloodV = (
        d["VLu"], d["VM"], d["VHe"], d["VK"], d["VBRb"], d["VBRt"], d["Vrest"], d["VS"], d["VP"], d["VIT"],
        d["VTb"], d["VTt"], d["VL"], d["VBloodV"]
    )
    QC, QM, QHe, QK, QBR, Qrest, QS, QP, QIT, QT, QL, GFR, PABR, PAT = (
        d["QC"], d["QM"], d["QHe"], d["QK"], d["QBR"], d["Qrest"], d["QS"], d["QP"], d["QIT"], d["QT"], d["QL"],
        d["GFR"], d["PABR"], d["PAT"]
    )

    (ALu, AM, AHe, AK, ABRb, ABRt, Arest, AS, AP, AGutLu, AI, ATb, ATt, AL, AV) = y

    CLu = ALu / VLu
    CVLu = CLu / p.PLu * p.Rbp
    CA = CVLu

    CM = AM / VM
    CVM = CM / p.PM * p.Rbp

    CHe = AHe / VHe
    CVHe = CHe / p.PHe * p.Rbp

    CK = AK / VK
    CVK = CK / p.PK * p.Rbp

    CBRb = ABRb / VBRb
    CBRt = ABRt / VBRt
    CVBR = CBRb
    PermBR = PABR * (CBRb - CBRt / p.PBR * p.Rbp)

    Crest = Arest / Vrest
    CVrest = Crest / p.Prest * p.Rbp

    CS = AS / VS
    CVS = CS / p.PS * p.Rbp

    CP = AP / VP
    CVP = CP / p.PP * p.Rbp

    CI = AI / VIT
    CVI = CI / p.PInt * p.Rbp

    CTb = ATb / VTb
    CTt = ATt / VTt
    CVT = CTb
    PermT = PAT * (CTb - CTt / p.PT * p.Rbp)

    CL = AL / VL
    CVL = CL / p.PL * p.Rbp

    CV = AV / VBloodV

    CL_liv = p.VmaxL_livC * VL / (p.Km_liv + (CVL / p.Rbp) * p.Fup)

    dydt = np.zeros_like(y)
    dydt[STATE.index("ALu")] = QC * (CV - (ALu / VLu) / p.PLu * p.Rbp)
    dydt[STATE.index("AM")] = QM * (CA - CVM)
    dydt[STATE.index("AHe")] = QHe * (CA - CVHe)
    dydt[STATE.index("AK")] = QK * (CA - CVK) - GFR * p.Fup * CK / p.PK
    dydt[STATE.index("ABRb")] = QBR * (CA - CVBR) - PermBR
    dydt[STATE.index("ABRt")] = PermBR
    dydt[STATE.index("Arest")] = Qrest * (CA - CVrest)
    dydt[STATE.index("AS")] = QS * (CA - CVS)
    dydt[STATE.index("AP")] = QP * (CA - CVP)
    dydt[STATE.index("AGutLu")] = -p.Ka * AGutLu
    dydt[STATE.index("AI")] = QIT * (CA - CVI) + p.Ka * AGutLu * p.Fab
    dydt[STATE.index("ATb")] = QT * (CA - CVT) - PermT
    dydt[STATE.index("ATt")] = PermT
    dydt[STATE.index("AL")] = (QL - QIT - QS - QP) * CA + QS * CVS + QP * CVP + QIT * CVI - QL * CVL - CL_liv * ((CVL / p.Rbp) * p.Fup)
    dydt[STATE.index("AV")] = (QBR * CVBR + QM * CVM + QHe * CVHe + QK * CVK + Qrest * CVrest + QL * CVL + QT * CVT) - QC * CV
    return dydt

def results_dataframe(t: np.ndarray, y: np.ndarray, p: Params, d: Dict[str, float]) -> pd.DataFrame:
    df = pd.DataFrame(y, columns=STATE)
    df.insert(0, "time_h", t)

    CV = df["AV"] / d["VBloodV"]
    Cplasma_mgL = CV / p.Rbp
    df["Plasma (ng/mL)"] = Cplasma_mgL * 1000.0

    df["Liver (ng/mL eq)"] = (df["AL"] / d["VL"]) * 1000.0
    df["Kidney (ng/mL eq)"] = (df["AK"] / d["VK"]) * 1000.0
    df["Muscle (ng/mL eq)"] = (df["AM"] / d["VM"]) * 1000.0
    df["Brain tissue (ng/mL eq)"] = (df["ABRt"] / d["VBRt"]) * 1000.0

    df["Tumor tissue (ng/mL eq)"] = (df["ATt"] / d["VTt"]) * 1000.0
    df["Tumor total (ng/mL eq)"] = ((df["ATb"] + df["ATt"]) / d["VT"]) * 1000.0
    return df

def auc_trapz(t: np.ndarray, c: np.ndarray) -> float:
    return float(np.trapz(c, t))

def pk_metrics_window(df: pd.DataFrame, series: str, t0: float, t1: float) -> Optional[Dict[str, float]]:
    sub = df[(df.time_h >= t0) & (df.time_h <= t1)].copy()
    if len(sub) < 2:
        return None
    t = sub.time_h.to_numpy()
    c = sub[series].to_numpy()
    cmax = float(np.max(c))
    tmax = float(t[np.argmax(c)] - t0)
    auc = float(np.trapz(c, t - t0))
    cmin = float(np.min(c))
    cavg = float(auc / (t1 - t0))
    ctau = float(c[-1])
    return {"Cmax": cmax, "Tmax": tmax, "AUC": auc, "Cavg": cavg, "Cmin": cmin, "Ctau": ctau}

def build_regimen(schedule_type: str, ld_mg: float, md_mg: float, days_total: int,
                 interval_h: float = 24.0, n_loading_doses: int = 1, loading_interval_h: float = 24.0,
                 start_maintenance_h: Optional[float] = None, custom_csv: str = "") -> List[Tuple[float, float]]:
    events: List[Tuple[float, float]] = []

    for i in range(max(1, int(n_loading_doses))):
        events.append((i * float(loading_interval_h), float(ld_mg)))

    if start_maintenance_h is None:
        start_maintenance_h = (max(1, int(n_loading_doses)) * float(loading_interval_h))

    t_end = float(days_total) * 24.0

    if schedule_type == "Single dose only":
        return [(0.0, float(ld_mg))]

    if schedule_type == "MWF (Mon/Wed/Fri)":
        dosing_dows = {0, 2, 4}
        for day in range(0, int(days_total) + 1):
            if day % 7 in dosing_dows:
                t = day * 24.0
                if t >= start_maintenance_h and t > 0:
                    events.append((t, float(md_mg)))
        return sorted(events)

    if schedule_type in ("Every X hours (qXh)", "Daily (q24h)", "BID (q12h)", "TID (q8h)", "Weekly (q168h)"):
        if schedule_type == "Daily (q24h)":
            interval_h = 24.0
        elif schedule_type == "BID (q12h)":
            interval_h = 12.0
        elif schedule_type == "TID (q8h)":
            interval_h = 8.0
        elif schedule_type == "Weekly (q168h)":
            interval_h = 168.0

        t = float(start_maintenance_h)
        while t <= t_end + 1e-9:
            events.append((t, float(md_mg)))
            t += float(interval_h)
        return sorted(events)

    if schedule_type == "Custom times (CSV: time_h,dose_mg)":
        ev: List[Tuple[float, float]] = []
        for line in custom_csv.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 2:
                continue
            try:
                ev.append((float(parts[0]), float(parts[1])))
            except ValueError:
                continue
        return sorted(ev)

    return sorted(events)

def simulate(p: Params, t_end_h: float, dose_events: List[Tuple[float, float]], dt_out_h: float = 0.1) -> pd.DataFrame:
    d = build_derived(p)
    y0 = np.zeros(len(STATE), dtype=float)

    dose_events = sorted(dose_events, key=lambda x: x[0])
    dose_map: Dict[float, float] = {}
    for t, dose in dose_events:
        dose_map.setdefault(float(t), 0.0)
        dose_map[float(t)] += float(dose)

    boundaries = sorted({0.0, *dose_map.keys(), float(t_end_h)})
    boundaries = [t for t in boundaries if 0.0 <= t <= t_end_h]
    if boundaries and boundaries[0] != 0.0:
        boundaries = [0.0] + boundaries
    if not boundaries:
        boundaries = [0.0, float(t_end_h)]
    if boundaries[-1] != t_end_h:
        boundaries = boundaries + [float(t_end_h)]

    t_all: List[float] = []
    y_all: List[np.ndarray] = []
    y = y0.copy()

    if 0.0 in dose_map:
        y[STATE.index("AGutLu")] += dose_map[0.0]

    for i in range(len(boundaries) - 1):
        t0 = boundaries[i]
        t1 = boundaries[i + 1]
        n = max(2, int(math.ceil((t1 - t0) / dt_out_h)) + 1)
        t_eval = np.linspace(t0, t1, n)

        sol = solve_ivp(lambda tt, yy: rhs(tt, yy, p, d), (t0, t1), y, method="LSODA",
                        rtol=1e-6, atol=1e-9, t_eval=t_eval)

        if not t_all:
            t_all.extend(sol.t.tolist())
            y_all.extend(sol.y.T)
        else:
            t_all.extend(sol.t[1:].tolist())
            y_all.extend(sol.y.T[1:])

        y = sol.y[:, -1].copy()

        if t1 in dose_map and t1 != t_end_h:
            y[STATE.index("AGutLu")] += dose_map[t1]

    return results_dataframe(np.asarray(t_all), np.asarray(y_all), p, d)

# ---------------- Sidebar UI ----------------
with st.sidebar:
    st.header("Dose regimen")
    schedule_type = st.selectbox(
        "Schedule type",
        ["MWF (Mon/Wed/Fri)", "Daily (q24h)", "BID (q12h)", "TID (q8h)",
         "Every X hours (qXh)", "Weekly (q168h)", "Custom times (CSV: time_h,dose_mg)", "Single dose only"]
    )
    days_total = st.number_input("Simulation horizon (days)", min_value=1, max_value=120, value=28, step=1)

    st.subheader("Loading + maintenance")
    ld_mg = st.number_input("Loading dose (mg)", min_value=0.0, value=24.0, step=0.5)
    n_ld = st.number_input("Number of loading doses", min_value=1, max_value=10, value=1, step=1)
    ld_int = st.number_input("Loading dose interval (h)", min_value=1.0, max_value=168.0, value=24.0, step=1.0)
    md_mg = st.number_input("Maintenance dose (mg)", min_value=0.0, value=8.0, step=0.5)

    if schedule_type == "Every X hours (qXh)":
        interval_h = st.number_input("Maintenance interval (h)", min_value=1.0, max_value=168.0, value=48.0, step=1.0)
    else:
        interval_h = 24.0

    maint_start = st.number_input(
        "Maintenance start time (h)",
        min_value=0.0, max_value=float(days_total) * 24.0,
        value=float(max(1, int(n_ld)) * float(ld_int)),
        step=1.0,
        help="Default starts after the last loading dose."
    )

    custom_csv = ""
    if schedule_type == "Custom times (CSV: time_h,dose_mg)":
        custom_csv = st.text_area("Dose events", value="0,24\n48,8\n96,8\n144,8\n192,8", height=140)

    st.divider()
    st.header("Model options")
    BW = st.number_input("Body weight (kg)", min_value=30.0, max_value=200.0, value=70.0, step=1.0)

    with st.expander("Advanced PK parameters", expanded=False):
        Fab = st.slider("Fab (fraction absorbed)", min_value=0.0, max_value=1.0, value=1.0, step=0.05)
        Ka = st.number_input("Ka (1/h)", min_value=0.001, max_value=5.0, value=0.297, step=0.01)
        Vmax = st.number_input("Vmax_livC (mg/h/kg organ wt)", min_value=0.01, max_value=10.0, value=0.760, step=0.05)
        Km = st.number_input("Km_liv (mg/L)", min_value=1e-5, max_value=1.0, value=0.0036, step=0.0005, format="%.4f")
        Fup = st.number_input("Fup", min_value=0.0001, max_value=0.2, value=0.008, step=0.001, format="%.4f")
        Rbp = st.number_input("Rbp", min_value=0.2, max_value=3.0, value=1.5, step=0.1)

    st.divider()
    st.header("Display")
    tissues = st.multiselect(
        "Curves to show",
        options=["Plasma (ng/mL)", "Tumor tissue (ng/mL eq)", "Tumor total (ng/mL eq)",
                 "Brain tissue (ng/mL eq)", "Liver (ng/mL eq)", "Kidney (ng/mL eq)", "Muscle (ng/mL eq)"],
        default=["Plasma (ng/mL)", "Tumor tissue (ng/mL eq)"]
    )
    y_log = st.checkbox("Log scale (y-axis)", value=False)

    st.divider()
    st.header("PK window")
    pk_window = st.selectbox("Compute PK metrics over", ["Whole simulation (0–t)", "Last dosing interval (tau)", "Custom window"])
    custom_t0 = st.number_input("Custom window start (h)", min_value=0.0, value=0.0, step=1.0)
    custom_t1 = st.number_input("Custom window end (h)", min_value=0.0, value=48.0, step=1.0)

# Build and run
p = Params(BW=float(BW), Fab=float(Fab), Ka=float(Ka), VmaxL_livC=float(Vmax),
           Km_liv=float(Km), Fup=float(Fup), Rbp=float(Rbp))

dose_events = build_regimen(schedule_type, float(ld_mg), float(md_mg), int(days_total),
                            float(interval_h), int(n_ld), float(ld_int), float(maint_start), custom_csv)

t_end_h = float(days_total) * 24.0

with st.spinner("Simulating PBPK model…"):
    df = simulate(p, t_end_h=t_end_h, dose_events=dose_events, dt_out_h=0.1)

tab_sim, tab_pk, tab_about = st.tabs(["Simulation", "PK metrics", "About"])

with tab_sim:
    left, right = st.columns([2.2, 1.0], gap="large")

    with left:
        st.subheader("Concentration–time profiles")
        fig = go.Figure()
        for series in tissues:
            if series in df.columns:
                fig.add_trace(go.Scatter(
                    x=df["time_h"], y=df[series],
                    mode="lines", name=series,
                    hovertemplate="t=%{x:.2f} h<br>%{y:.3g}<extra>" + series + "</extra>"
                ))
        fig.update_layout(
            height=520,
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            xaxis_title="Time (h)",
            yaxis_title="Concentration",
        )
        if y_log:
            fig.update_yaxes(type="log")
        st.plotly_chart(fig, use_container_width=True)

        st.caption("Tumor/brain/liver/kidney curves are **ng/mL-equivalent** (mg/L tissue × 1000) for convenience; not directly ng/g without an explicit conversion.")

    with right:
        st.subheader("Dose events")
        st.dataframe(pd.DataFrame(dose_events, columns=["time_h", "dose_mg"]), hide_index=True, use_container_width=True)

        st.subheader("Quick KPIs (plasma)")
        plasma = "Plasma (ng/mL)"
        c = df[plasma].to_numpy()
        t = df["time_h"].to_numpy()
        st.metric("Cmax (ng/mL)", f"{float(np.max(c)):.3g}")
        st.metric("Tmax (h)", f"{float(t[np.argmax(c)]):.3g}")
        st.metric("AUC0–t (ng·h/mL)", f"{auc_trapz(t, c):.3g}")

with tab_pk:
    st.subheader("PK indicators")
    series = st.selectbox("Series", options=[c for c in df.columns if c.endswith("ng/mL)") or c.endswith("ng/mL eq)")], index=0)

    tau = None
    if len(dose_events) >= 2:
        times = sorted([tt for tt, _ in dose_events if tt <= t_end_h])
        deltas = [times[i + 1] - times[i] for i in range(len(times) - 1) if times[i + 1] - times[i] > 1e-6]
        tau = deltas[-1] if deltas else None

    if pk_window == "Whole simulation (0–t)":
        t0, t1 = 0.0, float(t_end_h)
    elif pk_window == "Last dosing interval (tau)":
        if tau is None:
            st.warning("Not enough doses to infer tau. Using 0–t.")
            t0, t1 = 0.0, float(t_end_h)
        else:
            last_dose = max([tt for tt, _ in dose_events if tt <= t_end_h])
            t0, t1 = float(last_dose), float(min(t_end_h, last_dose + tau))
    else:
        t0, t1 = float(custom_t0), float(custom_t1)

    met = pk_metrics_window(df, series, t0, t1)
    if met is None:
        st.warning("Window has insufficient points.")
    else:
        cols = st.columns(6)
        cols[0].metric("Cmax", f"{met['Cmax']:.3g}")
        cols[1].metric("Tmax (h)", f"{met['Tmax']:.3g}")
        cols[2].metric("AUC", f"{met['AUC']:.3g}")
        cols[3].metric("Cavg", f"{met['Cavg']:.3g}")
        cols[4].metric("Cmin", f"{met['Cmin']:.3g}")
        cols[5].metric("Ctau", f"{met['Ctau']:.3g}")
        st.caption(f"Computed over **{t0:.2f}–{t1:.2f} h** for **{series}**.")

    st.divider()
    st.subheader("Export")
    st.download_button(
        "Download simulation CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="pbpk_simulation.csv",
        mime="text/csv",
        use_container_width=True
    )

with tab_about:
    st.subheader("Versatile simulations")
    st.markdown(
        """
**Regimen templates**
- MWF, daily, BID, TID, qXh, weekly
- Multiple loading doses + maintenance start time
- Custom CSV dosing (time_h,dose_mg)

**Tissues**
- Plasma
- Tumor tissue (ATt/VTt) and tumor total (ATb+ATt)/VT
- Brain tissue, liver, kidney, muscle (all as ng/mL-equivalent)

**PK metrics**
- Cmax, Tmax, AUC, Cavg, Cmin, Ctau over customizable windows
        """
    )
    st.subheader("Using the official DMPK Core logo")
    st.markdown(
        """
If you have the official **NCATS DMPK Core** logo file (PNG/SVG):
1. Add it to your GitHub repo (e.g., `assets/dmpk_core_logo.png`)
2. Replace the `st.image(...)` line near the top of `pbpk_app.py` with:
```python
st.image("assets/dmpk_core_logo.png", use_container_width=True)
```
        """
    )
