"""Shared look-and-feel and Plotly figure builders for the Streamlit dashboard."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

VIOLET = "#a78bfa"
TEAL = "#5eead4"
AMBER = "#f7b955"
ROSE = "#fb7185"
SKY = "#7dd3fc"
INK = "#f1eff8"
MUTED = "#a89fc4"
PANEL = "#1a1730"
BG = "#0f0d1a"
CATEGORICAL = [VIOLET, TEAL, AMBER, ROSE, SKY, "#c4b5fd"]
IMAGE_SCALE = [[0.0, "#0f0d1a"], [0.5, "#6d5bd0"], [1.0, "#f5f3f9"]]

CSS = f"""
<style>
.block-container {{ padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1400px; }}
h1, h2, h3 {{ letter-spacing: -0.01em; }}
.hero {{ background: linear-gradient(135deg, #1d1745 0%, #2a1c5c 55%, #12332f 140%);
         border: 1px solid #3a3266; border-radius: 18px; padding: 1.6rem 1.9rem; margin-bottom: 1rem; }}
.hero h1 {{ margin: 0 0 .25rem 0; font-size: 2.1rem; }}
.hero p {{ margin: 0; color: {MUTED}; font-size: 1.02rem; }}
.card {{ background: {PANEL}; border: 1px solid #332d49; border-radius: 14px; padding: 1rem 1.2rem; height: 100%; }}
.card h4 {{ margin: 0 0 .4rem 0; color: {VIOLET}; font-size: .95rem; text-transform: uppercase; letter-spacing: .06em; }}
.card p, .card li {{ color: #d9d4ea; font-size: .95rem; line-height: 1.5; }}
.kpi {{ background: {PANEL}; border: 1px solid #332d49; border-radius: 14px; padding: .85rem 1.1rem; }}
.kpi .v {{ font-size: 1.75rem; font-weight: 700; color: {INK}; line-height: 1.15; }}
.kpi .l {{ font-size: .78rem; text-transform: uppercase; letter-spacing: .07em; color: {MUTED}; }}
.kpi .s {{ font-size: .82rem; color: {TEAL}; }}
.pill {{ display: inline-block; padding: .15rem .6rem; border-radius: 999px; font-size: .8rem;
         background: #2b2557; color: #d8ccff; margin-right: .35rem; border: 1px solid #453b8a; }}
.step {{ color: {TEAL}; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; font-size: .8rem; }}
.callout {{ border-left: 4px solid {AMBER}; background: #2a2316; border-radius: 8px; padding: .7rem 1rem; color: #f3e6c4; }}
.good {{ border-left: 4px solid {TEAL}; background: #112622; border-radius: 8px; padding: .7rem 1rem; color: #cdf5ee; }}
.stTabs [data-baseweb="tab-list"] {{ gap: 4px; flex-wrap: wrap; }}
.stTabs [data-baseweb="tab"] {{ background: {PANEL}; border-radius: 10px 10px 0 0; padding: .45rem .9rem; }}
.stTabs [aria-selected="true"] {{ background: #2b2557 !important; }}
</style>
"""


def kpi(label: str, value: str, sub: str = "") -> str:
    sub_html = f'<div class="s">{sub}</div>' if sub else ""
    return f'<div class="kpi"><div class="l">{label}</div><div class="v">{value}</div>{sub_html}</div>'


def card(title: str, body_html: str) -> str:
    return f'<div class="card"><h4>{title}</h4>{body_html}</div>'


def style_fig(fig: go.Figure, height: int = 360, title: str | None = None) -> go.Figure:
    fig.update_layout(
        template="plotly_dark", height=height, paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=10, r=10, t=44 if title else 16, b=10),
        font=dict(color=INK), legend=dict(orientation="h", y=1.12, x=0),
        title=dict(text=title, x=0.01) if title else None,
    )
    fig.update_xaxes(gridcolor="#2a2547", zeroline=False)
    fig.update_yaxes(gridcolor="#2a2547", zeroline=False)
    return fig


def pixel_heatmap(arr: np.ndarray, title: str | None = None, annotate: bool = True,
                  height: int = 380, zmax: float | None = None) -> go.Figure:
    arr = np.asarray(arr)
    size = arr.shape[0]
    text = [[f"{v:.2f}" for v in row] for row in arr] if annotate and size <= 8 else None
    fig = go.Figure(go.Heatmap(
        z=arr, colorscale=IMAGE_SCALE, zmin=0, zmax=zmax if zmax is not None else max(1e-9, float(arr.max())),
        text=text, texttemplate="%{text}" if text else None, textfont=dict(size=12 if size <= 4 else 10),
        showscale=False, hovertemplate="row %{y}, col %{x}: %{z:.3f}<extra></extra>",
    ))
    labels = size <= 8
    fig.update_yaxes(autorange="reversed", scaleanchor="x", showgrid=False, dtick=1, showticklabels=labels)
    fig.update_xaxes(showgrid=False, dtick=1, side="top", showticklabels=labels)
    return style_fig(fig, height, title)


def basis_state_bars(probs: np.ndarray, labels: list[str], colours: list[str], title: str,
                     height: int = 340) -> go.Figure:
    fig = go.Figure(go.Bar(x=labels, y=probs, marker_color=colours,
                           hovertemplate="|%{x}⟩  P = %{y:.4f}<extra></extra>"))
    fig.update_xaxes(tickangle=-60, tickfont=dict(size=9))
    fig.update_yaxes(title="probability")
    return style_fig(fig, height, title)


def confusion_heatmap(cm: np.ndarray, class_names: list[str], title: str, height: int = 330) -> go.Figure:
    fig = go.Figure(go.Heatmap(
        z=cm, x=[f"pred {c}" for c in class_names], y=[f"true {c}" for c in class_names],
        colorscale=[[0, "#17142b"], [1, VIOLET]], text=cm, texttemplate="%{text}",
        textfont=dict(size=16), showscale=False,
    ))
    fig.update_yaxes(autorange="reversed")
    return style_fig(fig, height, title)


def history_figures(history, metric_name: str = "Accuracy") -> tuple[go.Figure, go.Figure]:
    epochs = list(range(1, len(history.train_loss) + 1))
    loss = go.Figure()
    loss.add_trace(go.Scatter(x=epochs, y=history.train_loss, name="train", line=dict(color=VIOLET, width=3)))
    loss.add_trace(go.Scatter(x=epochs, y=history.val_loss, name="held-out", line=dict(color=TEAL, width=3)))
    loss.update_xaxes(title="epoch")
    loss.update_yaxes(title="loss")
    acc = go.Figure()
    acc.add_trace(go.Scatter(x=epochs, y=history.train_metric, name="train", line=dict(color=VIOLET, width=3)))
    acc.add_trace(go.Scatter(x=epochs, y=history.val_metric, name="held-out", line=dict(color=TEAL, width=3)))
    acc.update_xaxes(title="epoch")
    acc.update_yaxes(title=metric_name.lower(), range=[0, 1.02])
    return style_fig(loss, 330, "Loss"), style_fig(acc, 330, metric_name)


def sweep_line(df, x: str, y: str, title: str, err: str | None = None, xlabel: str | None = None,
               log_x: bool = False, height: int = 340) -> go.Figure:
    fig = go.Figure()
    kwargs = {}
    if err and err in df and df[err].abs().sum() > 0:
        kwargs["error_y"] = dict(type="data", array=df[err], color=MUTED)
    fig.add_trace(go.Scatter(x=df[x].astype(str) if df[x].dtype == object else df[x], y=df[y],
                             mode="lines+markers", line=dict(color=VIOLET, width=3),
                             marker=dict(size=10, color=TEAL), **kwargs))
    fig.update_xaxes(title=xlabel or x, type="log" if log_x else None)
    fig.update_yaxes(title=y)
    return style_fig(fig, height, title)


def pipeline_dot(encoding_label: str, n_qubits: int, n_layers: int) -> str:
    return f"""
digraph G {{
  rankdir=LR; bgcolor="transparent"; nodesep=0.35; ranksep=0.45;
  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=18, color="#453b8a",
        fillcolor="#1f1b3d", fontcolor="#f1eff8", margin="0.22,0.14"];
  edge [color="#7c6fd0", arrowsize=0.8];
  img   [label="Image\\n(upload / camera)"];
  pre   [label="Preprocess\\ngray · crop · resize\\n· normalise"];
  enc   [label="Quantum encoding\\n{encoding_label}", fillcolor="#2b2557", color="#a78bfa"];
  ans   [label="Variational circuit\\n{n_layers} layers on {n_qubits} qubits", fillcolor="#2b2557", color="#a78bfa"];
  meas  [label="Measure ⟨Z⟩\\non each qubit", fillcolor="#12332f", color="#5eead4"];
  head  [label="Classical head\\n(linear + softmax)"];
  out   [label="Class\\nprediction", fillcolor="#2a2316", color="#f7b955"];
  img -> pre -> enc -> ans -> meas -> head -> out;
}}
"""


def circuit_blocks_dot(encoding_label: str, n_qubits: int, n_layers: int, n_classes: int) -> str:
    return f"""
digraph G {{
  rankdir=LR; bgcolor="transparent"; nodesep=0.3; ranksep=0.4;
  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=18, color="#453b8a",
        fillcolor="#1f1b3d", fontcolor="#f1eff8", margin="0.22,0.14"];
  edge [color="#7c6fd0"];
  x    [label="pixels x\\n(classical data)", fillcolor="#17142b"];
  init [label="|0⟩ ⊗ {n_qubits}\\ninitial state"];
  enc  [label="U_enc(x)\\n{encoding_label}\\n(fixed, no trainable params)", fillcolor="#2b2557", color="#a78bfa"];
  var  [label="U(θ) × {n_layers}\\nrotations + CNOT entanglers\\n(trainable θ)", fillcolor="#3a2a6e", color="#c4b5fd"];
  meas [label="⟨Z_0⟩ … ⟨Z_{n_qubits - 1}⟩\\n(expectation values)", fillcolor="#12332f", color="#5eead4"];
  head [label="Linear layer\\n{n_qubits} → {1 if n_classes <= 2 else n_classes} (trainable)"];
  y    [label="logits → class", fillcolor="#2a2316", color="#f7b955"];
  x -> enc; init -> enc -> var -> meas -> head -> y;
}}
"""
