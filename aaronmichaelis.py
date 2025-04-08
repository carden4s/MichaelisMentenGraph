import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy.optimize import curve_fit
from scipy import stats
import statsmodels.api as sm


S = np.array([80, 100, 150, 200, 250, 300], dtype=float)  # [S] en mM
Vo = np.array([0.007066, 0.007519254, 0.010251723, 
               0.011321443, 0.01313966356, 0.015464126], dtype=float)  # Vo en mM/seg


def michaelis_menten(s, Vmax, Km):
    return (Vmax * s) / (Km + s)

# Ajuste no lineal con estimación robusta
Vmax_initial = np.max(Vo) * 1.2
Km_initial = S[np.argmax(Vo)] * 0.8
p0 = [Vmax_initial, Km_initial]

popt, pcov = curve_fit(michaelis_menten, S, Vo, p0=p0, maxfev=2000)
Vmax, Km = popt
Vmax_err, Km_err = np.sqrt(np.diag(pcov))

# Intervalos de confianza al 95%
t_critical = stats.t.ppf(0.975, len(S)-2)
Vmax_ci = (Vmax - t_critical * Vmax_err, Vmax + t_critical * Vmax_err)
Km_ci = (Km - t_critical * Km_err, Km + t_critical * Km_err)

# ==================================================================
# VALIDACIÓN ESTADÍSTICA
# ==================================================================
Vo_pred = michaelis_menten(S, Vmax, Km)
residuals = Vo - Vo_pred

rmse = np.sqrt(np.mean(residuals**2))
r_squared = 1 - np.sum(residuals**2)/np.sum((Vo - np.mean(Vo))**2)
adj_r_squared = 1 - (1 - r_squared)*(len(S)-1)/(len(S)-2)

shapiro_stat, shapiro_p = stats.shapiro(residuals)
dw = sm.stats.stattools.durbin_watson(residuals)

# Bootstrap para Km
n_boot = 2000
Km_values = []
for _ in range(n_boot):
    indices = np.random.choice(range(len(S)), size=len(S), replace=True)
    try:
        popt_boot, _ = curve_fit(michaelis_menten, S[indices], Vo[indices], p0=p0)
        Km_values.append(popt_boot[1])
    except RuntimeError:
        continue
Km_values = np.array(Km_values)

# ==================================================================
# VISUALIZACIÓN MEJORADA
# ==================================================================
fig = make_subplots(
    rows=3, cols=2,
    specs=[
        [{"type": "xy", "colspan": 2}, None],
        [{"type": "xy"}, {"type": "xy"}],
        [{"type": "table", "colspan": 2}, None]
    ],
    subplot_titles=(
        'Curva de Michaelis-Menten',
        'Análisis de Residuales (±RMSE)',
        'Distribución Bootstrap de Km',
        'Resultados Estadísticos'
    ),
    vertical_spacing=0.12
)

# Gráfico principal
S_fit = np.linspace(50, 350, 200)
fig.add_trace(
    go.Scatter(
        x=S_fit, y=michaelis_menten(S_fit, Vmax, Km),
        mode='lines',
        name='Modelo',
        line=dict(color='#e74c3c', width=3)
    ),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(
        x=S, y=Vo,
        mode='markers',
        name='Datos',
        marker=dict(
            color='#3498db',
            size=12,
            line=dict(width=1, color='#2c3e50')
        )
    ),
    row=1, col=1
)

# Residuales con banda de RMSE
fig.add_trace(
    go.Scatter(
        x=S, y=residuals,
        mode='markers',
        name='Residuales',
        marker=dict(color='#27ae60', size=10)
    ),
    row=2, col=1
)

fig.add_shape(
    type='rect',
    x0=S.min()-10, y0=-rmse,
    x1=S.max()+10, y1=rmse,
    fillcolor='#27ae60',
    opacity=0.1,
    line=dict(width=0),
    row=2, col=1
)

# Histograma de Km
fig.add_trace(
    go.Histogram(
        x=Km_values,
        nbinsx=25,
        marker=dict(color='#34495e', opacity=0.7),
        name='Distribución Km',
        hovertemplate="Km: %{x:.1f} mM<br>Frecuencia: %{y}"),
    row=2, col=2
)

# Tabla de resultados optimizada
stats_table = go.Table(
    header=dict(
        values=['<b>Parámetro</b>', '<b>Valor ± Error</b>', '<b>IC 95%</b>'],
        fill_color='#2c3e50',
        font=dict(color='white', size=14),
    ),
    cells=dict(
        values=[
            ['Vmax (mM/seg)', 'Km (mM)', 'R²', 'R² Ajustado', 'RMSE', 'Shapiro-Wilk (p)', 'Durbin-Watson'],
            [
                f"{Vmax:.5f} ± {Vmax_err:.5f}",
                f"{Km:.1f} ± {Km_err:.1f}",
                f"{r_squared:.4f}",
                f"{adj_r_squared:.4f}",
                f"{rmse:.6f}",
                f"{shapiro_p:.4f}",
                f"{dw:.2f}"
            ],
            [
                f"[{Vmax_ci[0]:.5f}, {Vmax_ci[1]:.5f}]",
                f"[{Km_ci[0]:.1f}, {Km_ci[1]:.1f}]",
                '-', '-', '-', '-', '-'
            ]
        ],
        fill_color=['#f8f9fa', '#ffffff', '#f8f9fa'],
        font=dict(color='#2c3e50', size=12),
    ),
    columnwidth=[1.5, 1.5, 2]
)

fig.add_trace(stats_table, row=3, col=1)

# Configuración final
fig.update_layout(
    height=1000,
    title_text="<b>Análisis Cinético Enzimático Completo</b>",
    title_font=dict(size=24, color='#2c3e50'),
    title_x=0.5,
    margin=dict(t=120, b=80, l=40, r=40),
    template='plotly_white',
    hoverlabel=dict(font_size=12)
)

fig.update_xaxes(
    showgrid=True, 
    gridcolor='#bdc3c7',
    linecolor='#2c3e50',
    title_font=dict(size=12)
)

fig.update_yaxes(
    showgrid=True,
    gridcolor='#bdc3c7',
    linecolor='#2c3e50',
    title_font=dict(size=12)
)

fig.show()