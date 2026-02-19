
import numpy as np
import pandas as pd
import streamlit as st
from dataclasses import dataclass
from scipy.integrate import solve_ivp

st.set_page_config(page_title="PBPK Dose Simulator", layout="wide")
st.title("PBPK Dose Simulator (Loading + Maintenance)")

@dataclass
class Params:
    VmaxL_livC: float = 0.760
    Km_liv: float = 0.0036
    Ka: float = 0.297
    Fab: float = 1.0
    Fup: float = 0.008
    Rbp: float = 1.5
    BW: float = 70.0
    VBloodVC: float = 0.044
    VLC: float = 0.019

STATE = ["AGut", "ALiver", "AVenous"]

def build_derived(p):
    VL = p.BW * p.VLC
    VVen = p.BW * p.VBloodVC
    return VL, VVen

def rhs(t, y, p):
    AGut, ALiver, AVen = y
    VL, VVen = build_derived(p)

    Ka = p.Ka
    CL_liv = p.VmaxL_livC * VL / (p.Km_liv + (ALiver/VL)*p.Fup)

    dAGut = -Ka * AGut
    dALiver = Ka * AGut - CL_liv * (ALiver/VL)
    dAVen = CL_liv * (ALiver/VL)

    return [dAGut, dALiver, dAVen]

def simulate(dose_events, days=7):
    p = Params()
    y0 = np.zeros(3)
    t_end = days * 24
    times = np.linspace(0, t_end, 1000)

    # apply loading dose at t=0
    y0[0] = dose_events[0][1]

    sol = solve_ivp(lambda t,y: rhs(t,y,p), [0, t_end], y0, t_eval=times)

    VL, VVen = build_derived(p)
    Cplasma = (sol.y[2] / VVen) * 1000

    df = pd.DataFrame({"time_h": sol.t, "Cplasma_ng_mL": Cplasma})
    return df

st.sidebar.header("Regimen")
ld = st.sidebar.number_input("Loading dose (mg)", value=24.0)
md = st.sidebar.number_input("Maintenance dose (mg)", value=8.0)
days = st.sidebar.number_input("Simulation days", value=14)

dose_events = [(0, ld)]
df = simulate(dose_events, days)

st.line_chart(df.set_index("time_h"))
st.write("Cmax (ng/mL):", df["Cplasma_ng_mL"].max())
st.write("AUC (ng·h/mL):", np.trapz(df["Cplasma_ng_mL"], df["time_h"]))
