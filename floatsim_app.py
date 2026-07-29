import streamlit as st
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import odeint
import matplotlib.pyplot as plt

st.title("FloatSim")
st.subheader("Simulador de Flotación de Arsenopirita Aurífera")

# ── SLIDERS ──────────────────────────────────────
st.sidebar.header("Parámetros de entrada")

feed        = st.sidebar.number_input("Feed (t/h)", 10, 500, 100)
ley_Au_feed = st.sidebar.number_input("Ley Au feed (g/t)", 1.0, 50.0, 10.0)
pct_As_feed = st.sidebar.number_input("As en feed (%)", 0.5, 15.0, 4.0)
k_Au        = st.sidebar.slider("k_Au (min⁻¹)", 0.01, 0.5, 0.08)
k_As        = st.sidebar.slider("k_As (min⁻¹)", 0.01, 0.3, 0.03)
pct_conc    = st.sidebar.slider("Fracción concentrado", 0.01, 0.15, 0.05)

Rmax_Au = 85
Rmax_As = 70
y0      = [0, 0]

# ── MODELO ───────────────────────────────────────
def modelo(y, t, k_Au, k_As, Rmax_Au, Rmax_As):
    R_Au, R_As = y
    return [k_Au*(Rmax_Au-R_Au), k_As*(Rmax_As-R_As)]

def calcular_R_Au(vars):
    t1, t2, t3 = vars
    s1 = odeint(modelo, y0, [0,t1], args=(k_Au,k_As,Rmax_Au,Rmax_As))
    s2 = odeint(modelo, s1[-1], [0,t2], args=(k_Au,k_As,Rmax_Au,Rmax_As))
    s3 = odeint(modelo, s2[-1], [0,t3], args=(k_Au,k_As,Rmax_Au,Rmax_As))
    return s3[-1][0]

def objetivo(vars):
    t1, t2, t3 = vars
    s1 = odeint(modelo, y0, [0,t1], args=(k_Au,k_As,Rmax_Au,Rmax_As))
    s2 = odeint(modelo, s1[-1], [0,t2], args=(k_Au,k_As,Rmax_Au,Rmax_As))
    s3 = odeint(modelo, s2[-1], [0,t3], args=(k_Au,k_As,Rmax_Au,Rmax_As))
    return -(s3[-1][0] - 1.5*s3[-1][1])

constraints = [
    {'type':'ineq','fun': lambda v: v[0]-5},
    {'type':'ineq','fun': lambda v: 15-v[0]},
    {'type':'ineq','fun': lambda v: v[1]-5},
    {'type':'ineq','fun': lambda v: 15-v[1]},
    {'type':'ineq','fun': lambda v: v[2]-5},
    {'type':'ineq','fun': lambda v: 15-v[2]},
    {'type':'ineq','fun': lambda v: calcular_R_Au(v)-70},
]

# ── CORRER ───────────────────────────────────────
st.info("Corriendo optimización...")
res = minimize(objetivo, x0=[8,4,2], constraints=constraints)
t1_opt, t2_opt, t3_opt = res.x

s1 = odeint(modelo, y0,     [0,t1_opt], args=(k_Au,k_As,Rmax_Au,Rmax_As))
s2 = odeint(modelo, s1[-1], [0,t2_opt], args=(k_Au,k_As,Rmax_Au,Rmax_As))
s3 = odeint(modelo, s2[-1], [0,t3_opt], args=(k_Au,k_As,Rmax_Au,Rmax_As))

R_Au_pct = s3[-1][0]
R_As_pct = s3[-1][1]

# ── MÉTRICAS ─────────────────────────────────────
st.success("Optimización completa")
col1, col2, col3, col4 = st.columns(4)
col1.metric("t1 óptimo", f"{t1_opt:.1f} min")
col2.metric("t2 óptimo", f"{t2_opt:.1f} min")
col3.metric("t3 óptimo", f"{t3_opt:.1f} min")
col4.metric("Selectividad F", f"{-res.fun:.2f}")

col5, col6 = st.columns(2)
col5.metric("Recuperación Au", f"{R_Au_pct:.1f}%")
col6.metric("Recuperación As", f"{R_As_pct:.1f}%")

# ── BALANCE DE MASA ───────────────────────────────
R_Au = R_Au_pct / 100
R_As = R_As_pct / 100

concentrado    = feed * pct_conc
relave         = feed - concentrado
Au_feed        = feed * ley_Au_feed
Au_conc        = Au_feed * R_Au
Au_relave      = Au_feed - Au_conc
ley_Au_conc    = Au_conc / concentrado
ley_Au_relave  = Au_relave / relave
As_feed        = feed * (pct_As_feed/100)
As_conc        = As_feed * R_As
As_relave      = As_feed - As_conc
pct_As_conc    = (As_conc / concentrado) * 100
pct_As_relave  = (As_relave / relave) * 100

import pandas as pd
st.subheader("Balance de Masa")
df = pd.DataFrame({
    'Corriente':   ['Feed', 'Concentrado', 'Relave'],
    'Flujo (t/h)': [feed, concentrado, relave],
    'Au (g/t)':    [ley_Au_feed, round(ley_Au_conc,1), round(ley_Au_relave,2)],
    'As (%)':      [pct_As_feed, round(pct_As_conc,1), round(pct_As_relave,1)],
})
st.dataframe(df, hide_index=True)

# ── GRÁFICA ───────────────────────────────────────
st.subheader("Cinética por Celda")
t1_c = np.linspace(0, t1_opt, 100)
t2_c = np.linspace(0, t2_opt, 100)
t3_c = np.linspace(0, t3_opt, 100)
s1g  = odeint(modelo, [0,0],     t1_c, args=(k_Au,k_As,Rmax_Au,Rmax_As))
s2g  = odeint(modelo, s1g[-1],   t2_c, args=(k_Au,k_As,Rmax_Au,Rmax_As))
s3g  = odeint(modelo, s2g[-1],   t3_c, args=(k_Au,k_As,Rmax_Au,Rmax_As))

fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
for i, (tc, sg, title) in enumerate([
    (t1_c, s1g, f'Celda 1 — {t1_opt:.1f} min'),
    (t2_c, s2g, f'Celda 2 — {t2_opt:.1f} min'),
    (t3_c, s3g, f'Celda 3 — {t3_opt:.1f} min'),
]):
    axes[i].plot(tc, sg[:,0], color='gold', lw=2, label='Au')
    axes[i].plot(tc, sg[:,1], color='gray', lw=2, ls='--', label='As')
    axes[i].set_title(title)
    axes[i].set_xlabel('Tiempo (min)')
    axes[i].grid(True, alpha=0.3)
    axes[i].legend()
axes[0].set_ylabel('Recuperación (%)')
plt.tight_layout()
st.pyplot(fig)
