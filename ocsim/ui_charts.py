"""
ui_charts.py — funções de gráfico (Plotly) usadas pela interface
=================================================================

PAPEL NO PROJETO
----------------
Centraliza o visual dos gráficos para que todos sigam o mesmo padrão:
* cada cenário tem SEMPRE a mesma cor (paleta fixa, testada para daltonismo);
* o log real aparece sempre em cinza escuro;
* grade discreta, fundo transparente, legenda no topo.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Paleta categórica (ordem fixa) — um cenário = uma cor, em todos os gráficos
# ---------------------------------------------------------------------------
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
POLICY_COLOR = {"ATUAL": SERIES[0], "SQ": SERIES[1], "RS": SERIES[2], "RSS": SERIES[3], "L4L": SERIES[4]}
LOG_COLOR = "#52514e"   # log real (referência)
MUTED = "#8a8984"       # linhas auxiliares (BWE = 1, barras de erro)
GRID = "#e6e5e0"        # grade
TEXT = "#0b0b0b"        # textos e rótulos


def style(fig: go.Figure, height: int = 380, title: str | None = None, ytitle: str | None = None,
          xtitle: str | None = None, legend: bool = True) -> go.Figure:
    """Aplica o estilo padrão a qualquer figura Plotly."""
    fig.update_layout(
        height=height,
        title=dict(text=title, x=0, font=dict(size=15, color=TEXT)) if title else None,
        margin=dict(l=10, r=10, t=50 if title else 20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, system-ui, sans-serif", size=12, color="#52514e"),
        hoverlabel=dict(bgcolor="white", font_size=12),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0),
        bargap=0.25,
    )
    fig.update_xaxes(showgrid=False, linecolor=GRID, title=xtitle, ticks="outside", tickcolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, title=ytitle)
    return fig


def color_for(policy: str, i: int = 0) -> str:
    """Cor de um cenário pela sua política (ou pela posição, se a política for desconhecida)."""
    return POLICY_COLOR.get(policy, SERIES[i % len(SERIES)])


def hist_with_fit(samples: np.ndarray, dist, title: str, xtitle: str, bins: int = 40) -> go.Figure:
    """Histograma das observações do log com a curva da distribuição ajustada por cima.
    Serve para julgar visualmente a qualidade do ajuste (etapa 3)."""
    x = np.asarray(samples, dtype=float)
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=x, nbinsx=bins, histnorm="probability density", name="observado (log)",
                               marker=dict(color=SERIES[0], line=dict(color="white", width=1)), opacity=0.85))
    if dist is not None and dist.kind == "parametric" and len(x) > 1:
        grid = np.linspace(np.nanmin(x), np.nanmax(x), 300)
        fig.add_trace(go.Scatter(x=grid, y=dist.pdf(grid), mode="lines", name=f"ajuste: {dist.name}",
                                 line=dict(color=SERIES[1], width=2)))
    return style(fig, title=title, xtitle=xtitle, ytitle="densidade")


def bar_ci(df: pd.DataFrame, value: str, lo: str, hi: str, cat: str, policies: list[str], title: str,
           ytitle: str, ref: float | None = None, ref_label: str = "log real") -> go.Figure:
    """Barras por cenário com a média, barra de erro do IC 95% e (opcional) a
    linha tracejada do valor medido no log real. Para gráficos de BWE também
    desenha a linha pontilhada BWE = 1 (sem amplificação).

    df: uma linha por cenário com as colunas ``cat`` (nome), ``value`` (média),
        ``lo`` e ``hi`` (limites do IC)."""
    fig = go.Figure()
    for i, (_, r) in enumerate(df.iterrows()):
        fig.add_trace(go.Bar(
            x=[r[cat]], y=[r[value]], name=r[cat], marker=dict(color=color_for(policies[i], i), cornerradius=4),
            error_y=dict(type="data", symmetric=False, array=[max(r[hi] - r[value], 0)],
                         arrayminus=[max(r[value] - r[lo], 0)], color=MUTED, thickness=1.5, width=6),
            hovertemplate=f"%{{x}}<br>{ytitle}: %{{y:.3g}}<br>IC95%: [{r[lo]:.3g}; {r[hi]:.3g}]<extra></extra>",
        ))
        # rótulo do valor acima da barra de erro (para não sobrepor)
        top = r[hi] if np.isfinite(r[hi]) else r[value]
        fig.add_annotation(x=r[cat], y=top, text=f"<b>{r[value]:.3g}</b>", showarrow=False, yshift=12,
                           font=dict(color=TEXT, size=12))
    if ref is not None and np.isfinite(ref):
        fig.add_hline(y=ref, line=dict(color=LOG_COLOR, dash="dash", width=1.5),
                      annotation_text=f"{ref_label}: {ref:.3g}", annotation_position="top left",
                      annotation_font_color=LOG_COLOR, annotation_bgcolor="rgba(252,252,251,0.85)")
    if "BWE" in ytitle:
        fig.add_hline(y=1, line=dict(color=MUTED, dash="dot", width=1))
    return style(fig, title=title, ytitle=ytitle, legend=False)
