"""QNN Studio design system: theme, icons and small HTML components shared by every page."""
from __future__ import annotations

import html as _html

import plotly.graph_objects as go
import streamlit as st

VIOLET, TEAL, AMBER, ROSE, GREEN, SKY = "#8b7cf6", "#2dd4bf", "#f5b84a", "#fb7185", "#34d399", "#7dd3fc"
TEXT, MUTED, BG, SURFACE = "#ecebf8", "#9a96c0", "#0a0918", "#12102a"
COLORWAY = [VIOLET, TEAL, AMBER, ROSE, SKY, "#c4b5fd"]
MODEL_COLORS = {"Quantum neural network": VIOLET}

_ICONS = {
    "check": '<polyline points="20 6 9 17 4 12"/>',
    "alert": '<path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>',
    "upload": '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/>',
    "layers": '<polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/>',
    "bars": '<line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/>',
    "book": '<path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/>',
    "cpu": '<rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><line x1="9" y1="1" x2="9" y2="4"/><line x1="15" y1="1" x2="15" y2="4"/><line x1="9" y1="20" x2="9" y2="23"/><line x1="15" y1="20" x2="15" y2="23"/><line x1="20" y1="9" x2="23" y2="9"/><line x1="20" y1="14" x2="23" y2="14"/><line x1="1" y1="9" x2="4" y2="9"/><line x1="1" y1="14" x2="4" y2="14"/>',
    "sliders": '<line x1="4" y1="21" x2="4" y2="14"/><line x1="4" y1="10" x2="4" y2="3"/><line x1="12" y1="21" x2="12" y2="12"/><line x1="12" y1="8" x2="12" y2="3"/><line x1="20" y1="21" x2="20" y2="16"/><line x1="20" y1="12" x2="20" y2="3"/><line x1="1" y1="14" x2="7" y2="14"/><line x1="9" y1="8" x2="15" y2="8"/><line x1="17" y1="16" x2="23" y2="16"/>',
    "zap": '<polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>',
    "shield": '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>',
    "filter": '<polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3"/>',
    "target": '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>',
    "award": '<circle cx="12" cy="8" r="7"/><polyline points="8.21 13.89 7 23 12 20 17 23 15.79 13.88"/>',
    "play": '<polygon points="5 3 19 12 5 21 5 3"/>',
    "file": '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/>',
    "users": '<path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/>',
    "database": '<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/>',
    "flask": '<path d="M9 3h6"/><path d="M10 3v6L4.5 19a2 2 0 0 0 1.8 3h11.4a2 2 0 0 0 1.8-3L14 9V3"/>',
    "grad": '<path d="M22 10L12 5 2 10l10 5 10-5z"/><path d="M6 12v5c3 3 9 3 12 0v-5"/>',
    "atom": '<circle cx="12" cy="12" r="1.5"/><ellipse cx="12" cy="12" rx="10" ry="4"/><ellipse cx="12" cy="12" rx="10" ry="4" transform="rotate(60 12 12)"/><ellipse cx="12" cy="12" rx="10" ry="4" transform="rotate(120 12 12)"/>',
    "clock": '<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>',
    "sparkle": '<path d="M12 3l1.9 5.1L19 10l-5.1 1.9L12 17l-1.9-5.1L5 10l5.1-1.9z"/><path d="M19 15l.8 2.2L22 18l-2.2.8L19 21l-.8-2.2L16 18l2.2-.8z"/>',
    "search": '<circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>',
    "download": '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/>',
    "lock": '<rect x="3" y="11" width="18" height="11" rx="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/>',
    "trend": '<polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/>',
}


def icon(name: str, size: int = 20, color: str = "currentColor") -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
            f'stroke="{color}" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round">{_ICONS[name]}</svg>')


CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root{--bg:#0a0918;--s1:#12102a;--s2:#1a1738;--bd:rgba(167,139,250,.20);--tx:#ecebf8;--mu:#9a96c0;--v:#8b7cf6;--t:#2dd4bf;--a:#f5b84a;--r:#fb7185;--g:#34d399;}
html, body, .stApp, .stApp p, .stApp label, .stApp li, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp button, .stApp input, .stApp textarea, .stApp td, .stApp th {font-family:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;}
[data-testid="stIconMaterial"], .material-icons, .material-symbols-rounded, span[class*="material"]{font-family:'Material Symbols Rounded','Material Icons'!important;}
.stApp{background:radial-gradient(1100px 560px at 12% -8%, rgba(139,124,246,.20), transparent 60%),radial-gradient(900px 480px at 96% 2%, rgba(45,212,191,.10), transparent 55%),var(--bg);}
.block-container{max-width:1320px;padding-top:2.4rem;padding-bottom:4rem;}
h1,h2,h3,h4{letter-spacing:-.015em;font-weight:700;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#100e26 0%,#0c0b1f 100%);border-right:1px solid var(--bd);}
[data-testid="stHeader"]{background:rgba(10,9,24,.72);backdrop-filter:blur(10px);}
#MainMenu,footer{visibility:hidden;}

/* ---------- hero ---------- */
.q-hero{position:relative;overflow:hidden;border-radius:26px;padding:46px 48px 40px;border:1px solid var(--bd);
  background:linear-gradient(135deg,rgba(139,124,246,.26) 0%,rgba(91,141,239,.12) 45%,rgba(45,212,191,.10) 100%),var(--s1);
  box-shadow:0 30px 80px rgba(0,0,0,.35),inset 0 1px 0 rgba(255,255,255,.06);margin-bottom:26px;}
.q-hero:before{content:"";position:absolute;right:-90px;top:-90px;width:340px;height:340px;border-radius:50%;
  background:radial-gradient(circle,rgba(45,212,191,.28),transparent 68%);}
.q-hero:after{content:"";position:absolute;right:120px;bottom:-140px;width:300px;height:300px;border-radius:50%;
  background:radial-gradient(circle,rgba(139,124,246,.30),transparent 70%);}
.q-hero > *{position:relative;z-index:1;}
.q-eyebrow{font-size:.78rem;font-weight:700;letter-spacing:.14em;text-transform:uppercase;color:var(--t);margin-bottom:10px;}
.q-title{font-size:clamp(2rem,3.6vw,3.15rem);font-weight:800;line-height:1.08;margin:0 0 14px 0;
  background:linear-gradient(92deg,#fff 0%,#d9d2ff 55%,#7ee8d9 100%);-webkit-background-clip:text;background-clip:text;color:transparent;max-width:900px;}
.q-sub{font-size:1.08rem;line-height:1.6;color:#c9c5ea;max-width:820px;margin:0 0 20px 0;}
.q-chips{display:flex;flex-wrap:wrap;gap:8px;margin-top:6px;}
.q-chip{display:inline-flex;align-items:center;gap:6px;padding:5px 12px;border-radius:999px;font-size:.78rem;font-weight:600;
  background:rgba(139,124,246,.16);border:1px solid rgba(139,124,246,.38);color:#d9d2ff;}
.q-chip.t{background:rgba(45,212,191,.12);border-color:rgba(45,212,191,.35);color:#9af2e4;}
.q-chip.a{background:rgba(245,184,74,.12);border-color:rgba(245,184,74,.38);color:#fcd98f;}
.q-chip.r{background:rgba(251,113,133,.12);border-color:rgba(251,113,133,.38);color:#fdb4c0;}
.q-chip.g{background:rgba(52,211,153,.12);border-color:rgba(52,211,153,.35);color:#9cf0cc;}

/* ---------- sections ---------- */
.q-sec{margin:34px 0 16px 0;}
.q-sec .q-eyebrow{margin-bottom:4px;}
.q-sec h2{font-size:1.65rem;margin:0 0 6px 0;}
.q-sec p{color:var(--mu);margin:0;max-width:860px;line-height:1.6;}

/* ---------- cards ---------- */
.q-grid{display:grid;gap:16px;}
.q-grid.c2{grid-template-columns:repeat(2,1fr);} .q-grid.c3{grid-template-columns:repeat(3,1fr);} .q-grid.c4{grid-template-columns:repeat(4,1fr);}
@media (max-width:1000px){.q-grid.c3,.q-grid.c4{grid-template-columns:repeat(2,1fr);}} @media (max-width:640px){.q-grid.c2,.q-grid.c3,.q-grid.c4{grid-template-columns:1fr;}}
.q-card{background:linear-gradient(180deg,rgba(32,28,68,.78),rgba(22,19,50,.78));border:1px solid var(--bd);border-radius:20px;padding:22px 22px 20px;
  box-shadow:0 10px 34px rgba(0,0,0,.22);transition:transform .18s ease,border-color .18s ease,box-shadow .18s ease;height:100%;}
.q-card:hover{transform:translateY(-3px);border-color:rgba(139,124,246,.55);box-shadow:0 18px 44px rgba(0,0,0,.34);}
.q-card .q-ic{width:42px;height:42px;border-radius:13px;display:flex;align-items:center;justify-content:center;margin-bottom:14px;
  background:linear-gradient(135deg,rgba(139,124,246,.34),rgba(45,212,191,.18));border:1px solid rgba(139,124,246,.35);color:#e6e1ff;}
.q-card.t .q-ic{background:linear-gradient(135deg,rgba(45,212,191,.34),rgba(139,124,246,.14));border-color:rgba(45,212,191,.4);}
.q-card.a .q-ic{background:linear-gradient(135deg,rgba(245,184,74,.32),rgba(251,113,133,.14));border-color:rgba(245,184,74,.4);}
.q-card h4{margin:0 0 8px 0;font-size:1.08rem;color:#fff;}
.q-card p,.q-card li{color:#c9c5ea;font-size:.94rem;line-height:1.6;margin:0;}
.q-card ul{padding-left:18px;margin:8px 0 0 0;}
.q-card .q-foot{margin-top:14px;padding-top:12px;border-top:1px solid rgba(167,139,250,.14);color:var(--mu);font-size:.83rem;}
.q-badge{display:inline-block;font-size:.7rem;font-weight:800;letter-spacing:.1em;text-transform:uppercase;padding:4px 10px;border-radius:8px;margin-bottom:12px;}
.q-badge.research{background:rgba(139,124,246,.2);color:#d9d2ff;border:1px solid rgba(139,124,246,.45);}
.q-badge.learn{background:rgba(45,212,191,.16);color:#9af2e4;border:1px solid rgba(45,212,191,.4);}

/* ---------- KPI ---------- */
.q-kpi{background:linear-gradient(180deg,rgba(32,28,68,.78),rgba(22,19,50,.78));border:1px solid var(--bd);border-radius:18px;padding:16px 18px;height:100%;}
.q-kpi .l{font-size:.72rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:var(--mu);}
.q-kpi .v{font-size:1.9rem;font-weight:800;color:#fff;line-height:1.15;margin-top:4px;}
.q-kpi .s{font-size:.82rem;color:#9af2e4;margin-top:2px;}
.q-kpi.warn .s{color:#fcd98f;} .q-kpi.bad .s{color:#fdb4c0;} .q-kpi.mute .s{color:var(--mu);}

/* ---------- stepper ---------- */
.q-steps{display:flex;flex-wrap:wrap;gap:8px;margin:4px 0 18px 0;}
.q-step{display:flex;align-items:center;gap:9px;padding:8px 14px 8px 8px;border-radius:999px;border:1px solid var(--bd);background:rgba(26,23,56,.6);color:var(--mu);font-size:.84rem;font-weight:600;}
.q-step .n{width:24px;height:24px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:.76rem;font-weight:800;background:rgba(167,139,250,.16);color:#d9d2ff;}
.q-step.done{color:#d9f7f0;border-color:rgba(45,212,191,.4);} .q-step.done .n{background:var(--t);color:#04211d;}
.q-step.active{color:#fff;border-color:rgba(139,124,246,.7);background:rgba(139,124,246,.2);} .q-step.active .n{background:var(--v);color:#fff;}

/* ---------- callouts ---------- */
.q-call{border-radius:16px;padding:14px 18px;margin:18px 0 10px 0;border:1px solid var(--bd);background:rgba(139,124,246,.1);color:#ddd8ff;line-height:1.55;font-size:.95rem;}
.q-call b{color:#fff;}
.q-call.good{background:rgba(52,211,153,.10);border-color:rgba(52,211,153,.35);color:#c9f7e3;}
.q-call.warn{background:rgba(245,184,74,.10);border-color:rgba(245,184,74,.38);color:#fde7b4;}
.q-call.bad{background:rgba(251,113,133,.10);border-color:rgba(251,113,133,.38);color:#fed0d8;}

/* ---------- score ring ---------- */
.q-ring{--p:70;width:132px;height:132px;border-radius:50%;display:flex;align-items:center;justify-content:center;
  background:conic-gradient(var(--c,#2dd4bf) calc(var(--p)*1%),rgba(167,139,250,.14) 0);position:relative;margin:auto;}
.q-ring:before{content:"";position:absolute;inset:10px;border-radius:50%;background:#14122c;}
.q-ring span{position:relative;font-size:2.1rem;font-weight:800;color:#fff;}
.q-ring small{position:relative;display:block;font-size:.68rem;letter-spacing:.1em;color:var(--mu);text-transform:uppercase;font-weight:700;text-align:center;margin-top:-4px;}

/* ---------- streamlit widgets ---------- */
.stTabs [data-baseweb="tab-list"]{gap:4px;border-bottom:1px solid var(--bd);flex-wrap:wrap;}
.stTabs [data-baseweb="tab"]{height:46px;padding:0 18px;border-radius:12px 12px 0 0;background:transparent;color:var(--mu);font-weight:600;font-size:.93rem;}
.stTabs [data-baseweb="tab"]:hover{color:#fff;background:rgba(139,124,246,.1);}
.stTabs [aria-selected="true"]{color:#fff!important;background:rgba(139,124,246,.16);}
.stTabs [data-baseweb="tab-highlight"]{background:linear-gradient(90deg,var(--v),var(--t));height:3px;border-radius:3px;}
.stButton>button,.stDownloadButton>button{border-radius:12px;font-weight:650;border:1px solid var(--bd);background:rgba(26,23,56,.8);color:#ecebf8;padding:.55rem 1.1rem;transition:all .15s ease;}
.stButton>button:hover,.stDownloadButton>button:hover{border-color:rgba(139,124,246,.7);color:#fff;transform:translateY(-1px);}
.stButton>button[kind="primary"],.stDownloadButton>button[kind="primary"]{background:linear-gradient(135deg,#7c6cf0 0%,#5b8def 100%);border:0;color:#fff;box-shadow:0 10px 26px rgba(124,108,240,.38);}
.stButton>button[kind="primary"]:hover{box-shadow:0 14px 32px rgba(124,108,240,.5);}
div[data-baseweb="select"]>div,.stTextInput input,.stNumberInput input{border-radius:12px!important;}
[data-testid="stDataFrame"]{border-radius:14px;overflow:hidden;border:1px solid var(--bd);}
[data-testid="stExpander"]{border:1px solid var(--bd);border-radius:16px;background:rgba(22,19,50,.5);}
[data-testid="stFileUploaderDropzone"]{border-radius:16px;border:1.5px dashed rgba(139,124,246,.5);background:rgba(139,124,246,.06);}
.stProgress>div>div>div>div{background:linear-gradient(90deg,var(--v),var(--t));}
[data-testid="stMetric"]{background:linear-gradient(180deg,rgba(32,28,68,.78),rgba(22,19,50,.78));border:1px solid var(--bd);border-radius:16px;padding:14px 16px;}
[data-testid="stPageLink"]{margin-top:14px;}
[data-testid="stPageLink"] a{border:1px solid var(--bd);border-radius:14px;padding:.65rem 1rem;background:rgba(139,124,246,.14);font-weight:650;}
[data-testid="stPageLink"] a:hover{border-color:rgba(139,124,246,.75);background:rgba(139,124,246,.26);}
.q-foot-bar{margin-top:56px;padding-top:18px;border-top:1px solid var(--bd);color:var(--mu);font-size:.82rem;display:flex;justify-content:space-between;flex-wrap:wrap;gap:8px;}

/* ---------- tables ---------- */
.q-table{width:100%;border-collapse:separate;border-spacing:0 8px;font-size:.93rem;}
.q-table th{font-size:.7rem;letter-spacing:.1em;text-transform:uppercase;color:var(--mu);font-weight:700;text-align:left;padding:0 14px 4px;}
.q-table td{padding:13px 14px;background:rgba(26,23,56,.72);color:#e4e1fb;border-top:1px solid var(--bd);border-bottom:1px solid var(--bd);}
.q-table td:first-child{border-left:1px solid var(--bd);border-radius:14px 0 0 14px;font-weight:650;}
.q-table td:last-child{border-right:1px solid var(--bd);border-radius:0 14px 14px 0;}
.q-table tr.me td{background:linear-gradient(90deg,rgba(139,124,246,.28),rgba(139,124,246,.10));border-color:rgba(139,124,246,.55);color:#fff;}
.q-bar{height:8px;border-radius:6px;background:rgba(167,139,250,.14);overflow:hidden;min-width:120px;}
.q-bar i{display:block;height:100%;border-radius:6px;background:linear-gradient(90deg,#8b7cf6,#2dd4bf);}
.q-bar.alt i{background:linear-gradient(90deg,#64748b,#94a3b8);}
.q-sev{display:inline-block;font-size:.68rem;font-weight:800;letter-spacing:.08em;text-transform:uppercase;padding:3px 9px;border-radius:7px;margin-right:8px;}
.q-sev.high{background:rgba(251,113,133,.18);color:#fdb4c0;border:1px solid rgba(251,113,133,.4);}
.q-sev.medium{background:rgba(245,184,74,.16);color:#fcd98f;border:1px solid rgba(245,184,74,.4);}
.q-sev.low{background:rgba(125,211,252,.14);color:#bde7fb;border:1px solid rgba(125,211,252,.35);}
.q-issue{display:flex;gap:6px;align-items:flex-start;padding:12px 14px;border-radius:14px;background:rgba(26,23,56,.6);border:1px solid var(--bd);margin-bottom:8px;line-height:1.5;color:#d8d4f5;font-size:.92rem;}
.q-issue b{color:#fff;} .q-issue small{color:var(--mu);display:block;margin-top:2px;}
.q-tl{border-left:2px solid rgba(139,124,246,.35);margin:6px 0 6px 10px;padding-left:18px;}
.q-tl .it{position:relative;margin-bottom:14px;color:#d8d4f5;font-size:.93rem;line-height:1.5;}
.q-tl .it:before{content:"";position:absolute;left:-25px;top:6px;width:10px;height:10px;border-radius:50%;background:var(--t);box-shadow:0 0 0 4px rgba(45,212,191,.18);}
.q-tl .it b{color:#fff;display:block;}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def md(html: str) -> None:
    st.markdown(html, unsafe_allow_html=True)


def esc(text: str) -> str:
    return _html.escape(str(text))


def chip(text: str, tone: str = "") -> str:
    return f'<span class="q-chip {tone}">{text}</span>'


def hero(eyebrow: str, title: str, subtitle: str, chips: list[str] | None = None) -> None:
    chips_html = f'<div class="q-chips">{"".join(chips)}</div>' if chips else ""
    md(f'<div class="q-hero"><div class="q-eyebrow">{eyebrow}</div><h1 class="q-title">{title}</h1>'
       f'<p class="q-sub">{subtitle}</p>{chips_html}</div>')


def section(eyebrow: str, title: str, sub: str = "") -> None:
    sub_html = f"<p>{sub}</p>" if sub else ""
    md(f'<div class="q-sec"><div class="q-eyebrow">{eyebrow}</div><h2>{title}</h2>{sub_html}</div>')


def card(title: str, body: str, ic: str | None = None, tone: str = "", foot: str = "", badge: str = "", min_h: int = 0) -> str:
    ic_html = f'<div class="q-ic">{icon(ic, 22)}</div>' if ic else ""
    foot_html = f'<div class="q-foot">{foot}</div>' if foot else ""
    style = f' style="min-height:{min_h}px"' if min_h else ""
    return f'<div class="q-card {tone}"{style}>{badge}{ic_html}<h4>{title}</h4>{body}{foot_html}</div>'


def grid(cards: list[str], cols: int = 3) -> None:
    md(f'<div class="q-grid c{cols}">{"".join(cards)}</div>')


def kpi(label: str, value: str, sub: str = "", tone: str = "") -> str:
    sub_html = f'<div class="s">{sub}</div>' if sub else ""
    return f'<div class="q-kpi {tone}"><div class="l">{label}</div><div class="v">{value}</div>{sub_html}</div>'


def kpis(items: list[tuple]) -> None:
    cols = st.columns(len(items))
    for col, it in zip(cols, items):
        col.markdown(kpi(*it), unsafe_allow_html=True)


def stepper(steps: list[tuple[str, str]]) -> None:
    """steps: [(label, state)] with state in done | active | todo."""
    parts = []
    for i, (label, state) in enumerate(steps, 1):
        mark = icon("check", 14, "#04211d") if state == "done" else str(i)
        parts.append(f'<div class="q-step {state}"><span class="n">{mark}</span>{label}</div>')
    md(f'<div class="q-steps">{"".join(parts)}</div>')


def callout(kind: str, text: str) -> None:
    md(f'<div class="q-call {kind}">{text}</div>')


def score_ring(value: int, label: str = "Data quality") -> str:
    colour = "#34d399" if value >= 85 else ("#f5b84a" if value >= 65 else "#fb7185")
    return (f'<div class="q-ring" style="--p:{value};--c:{colour}"><div><span>{value}</span>'
            f'<small>{label}</small></div></div>')


def footer() -> None:
    md('<div class="q-foot-bar"><span><b>QNN Studio</b> · a no-code learning and benchmarking workspace for quantum neural networks</span>'
       '<span>Runs on a quantum <i>simulator</i>. Results are for learning and research, not for real-world decisions.</span></div>')


def style_fig(fig: go.Figure, height: int = 360, title: str | None = None, legend: bool = True) -> go.Figure:
    fig.update_layout(
        template="plotly_dark", height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=46 if title else 14, b=92 if legend else 10), colorway=COLORWAY,
        font=dict(family="Inter, -apple-system, Segoe UI, sans-serif", color=TEXT, size=13),
        title=dict(text=title, x=0.01, font=dict(size=15)) if title else None,
        legend=dict(orientation="h", yanchor="top", y=-0.3, x=0, bgcolor="rgba(0,0,0,0)") if legend else None, showlegend=legend,
        hoverlabel=dict(bgcolor="#1a1738", bordercolor="#8b7cf6", font=dict(color="#fff")))
    fig.update_xaxes(gridcolor="rgba(167,139,250,.10)", zerolinecolor="rgba(167,139,250,.18)", linecolor="rgba(167,139,250,.2)")
    fig.update_yaxes(gridcolor="rgba(167,139,250,.10)", zerolinecolor="rgba(167,139,250,.18)", linecolor="rgba(167,139,250,.2)")
    return fig
