"""Learning Lab: seven short, hands-on lessons that teach how a quantum neural network works."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from qnn_platform import ui
from qnn_platform.classical_model import baseline_label
from qnn_platform.cleaning import CleaningConfig, FittedCleaner, clean_structural, profile_dataframe
from qnn_platform.demo_data import lesson_messy_table, load_demo
from qnn_platform.figures import circuit_png, n_circuit_params
from qnn_platform.lab_content import GLOSSARY, LESSONS, TAKEAWAYS
from qnn_platform.workflow import (QNN_NAME, ModelConfig, PrepConfig, TrainConfig, metric_label, prepare_split, run_study,
                                   train_qnn, verdict)

st.session_state.setdefault("lab_done", set())
DONE: set = st.session_state["lab_done"]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def lesson_header(idx: int, lesson_id: str):
    _, title, minutes, goal = LESSONS[idx]
    status = ui.chip("Completed", "g") if lesson_id in DONE else ui.chip(f"About {minutes} min", "t")
    ui.md(f'<div class="q-sec" style="margin-top:6px"><div class="q-eyebrow">Lesson {idx + 1} of {len(LESSONS)}</div>'
          f'<h2>{title}</h2><p>{goal}</p><div class="q-chips" style="margin-top:10px">{status}</div></div>')


def quiz(lesson: str, qid: str, question: str, options: list[str], correct: int, why: str, mark: bool = True):
    key = f"quiz_{lesson}_{qid}"
    st.markdown(f"**{question}**")
    choice = st.radio("Choose one", options, index=None, key=key, label_visibility="collapsed")
    if st.button("Check my answer", key=key + "_btn"):
        if choice is None:
            st.session_state[key + "_res"] = ("none", "")
        elif options.index(choice) == correct:
            st.session_state[key + "_res"] = ("ok", why)
            if mark:
                DONE.add(lesson)
                st.rerun()
        else:
            st.session_state[key + "_res"] = ("no", "Not quite. Re-read the lesson above, then try again.")
    res = st.session_state.get(key + "_res")
    if res:
        kind, msg = res
        if kind == "ok":
            ui.callout("good", f"<b>Correct.</b> {msg}")
        elif kind == "no":
            ui.callout("warn", msg)
        else:
            ui.callout("", "Pick an answer first.")


def explain_cards(items):
    ui.grid([ui.card(t, f"<p>{b}</p>", ic=i, tone=tone) for t, b, i, tone in items], cols=len(items))
    ui.md("<div style='height:18px'></div>")


@st.cache_data(show_spinner=False)
def lr_experiment(lrs: tuple, epochs: int):
    demo = load_demo("iris")
    split = prepare_split(demo.df, PrepConfig(target=demo.target, n_qubits=4))
    out = {}
    for lr in lrs:
        r = train_qnn(split, ModelConfig(n_layers=2), TrainConfig(epochs=epochs, batch_size=16, lr=lr), seed=0)
        out[lr] = {"loss": list(r["history"].train_loss), "acc": float(r["history"].val_metric[-1])}
    return out


@st.cache_data(show_spinner=False)
def compare_lab(key: str, runs: int, epochs: int):
    demo = load_demo(key)
    res = run_study(demo.df, PrepConfig(target=demo.target, task=demo.task, n_qubits=4), ModelConfig(),
                    TrainConfig(epochs=epochs, batch_size=16, lr=0.05 if demo.task == "regression" else 0.1),
                    n_seeds=runs, baselines=["logistic_linear", "random_forest"])
    v, text = verdict(res)
    return res.summary, res.per_seed, res.n_val, v, text, res.primary


# --------------------------------------------------------------------------- #
# header, sidebar, progress
# --------------------------------------------------------------------------- #
ui.hero("Learning lab",
        "Learn how a quantum neural network works, hands on",
        "Seven short lessons in plain English. Every lesson has a small experiment you can run and a quick check at the end. "
        "No quantum background and no coding needed.",
        [ui.chip("Beginner friendly", "t"), ui.chip("About 70 minutes", ""), ui.chip("Runs on a simulator", "a")])
ui.stepper([(f"Lesson {i + 1}", "done" if l[0] in DONE else "todo") for i, l in enumerate(LESSONS)])

with st.sidebar:
    st.markdown("### Your progress")
    st.progress(len(DONE) / len(LESSONS), text=f"{len(DONE)} of {len(LESSONS)} lessons complete")
    for i, l in enumerate(LESSONS):
        mark = "✓" if l[0] in DONE else "○"
        st.markdown(f"<span style='color:{'#34d399' if l[0] in DONE else '#9a96c0'};font-weight:700'>{mark}</span> "
                    f"<span style='font-size:.88rem'>{i + 1}. {l[1]}</span> <span style='color:#9a96c0;font-size:.78rem'>· {l[2]} min</span>",
                    unsafe_allow_html=True)
    st.divider()
    if st.button("Reset my progress", width="stretch"):
        DONE.clear()
        for k in [k for k in st.session_state if k.startswith(("quiz_", "l1_", "l5_", "l6_", "l7_"))]:
            del st.session_state[k]
        st.rerun()
    st.caption("A lesson is complete when you answer its main question correctly.")

tabs = st.tabs([f"{i + 1}  {l[1]}" for i, l in enumerate(LESSONS)] + ["Glossary"])

# =========================================================================== #
# L1 WHAT IS A QNN
# =========================================================================== #
with tabs[0]:
    lesson_header(0, "l1")
    explain_cards([
        ("A neural network", "A stack of layers of numbers (weights). Training nudges the weights until the answers are right.", "layers", ""),
        ("A qubit", "A quantum bit. A normal bit is 0 or 1. A qubit can be a <i>mix</i> of both, until you measure it and get one answer.", "atom", "t"),
        ("A QNN", "Data goes into qubits, a circuit with <i>adjustable angles</i> processes them, you measure, and training tunes the angles.", "cpu", "a"),
    ])
    st.markdown("#### Try it: measure a qubit")
    st.caption("A qubit's tilt decides how likely it is to read 1. Each measurement gives only 0 or 1, so you need many measurements (shots) to see the odds.")
    c1, c2 = st.columns([1, 1.5], gap="large")
    theta = c1.slider("Tilt of the qubit (angle θ, in radians)", 0.0, 3.14, 1.2, 0.05, key="l1_theta")
    shots = c1.select_slider("Number of measurements (shots)", [10, 100, 1000, 10000], value=100, key="l1_shots")
    p1 = float(np.sin(theta / 2) ** 2)
    if c1.button("Measure the qubit", type="primary", key="l1_go"):
        ones = int(np.random.default_rng().binomial(shots, p1))
        st.session_state["l1_res"] = (round(theta, 2), shots, ones)
    res = st.session_state.get("l1_res")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=["Reads 0", "Reads 1"], y=[1 - p1, p1], name="True probability", marker_color=ui.VIOLET))
    if res and res[0] == round(theta, 2) and res[1] == shots:
        fig.add_trace(go.Bar(x=["Reads 0", "Reads 1"], y=[1 - res[2] / res[1], res[2] / res[1]], name=f"Measured ({shots} shots)", marker_color=ui.TEAL))
    fig.update_layout(barmode="group")
    fig.update_yaxes(range=[0, 1], tickformat=".0%")
    c2.plotly_chart(ui.style_fig(fig, 300, f"Chance of reading 1 = {p1:.0%}"), width="stretch")
    if res and res[0] == round(theta, 2) and res[1] == shots:
        ui.callout("", f"You measured <b>{res[2]} ones out of {res[1]}</b> ({res[2] / res[1]:.0%}). The true chance is {p1:.0%}. "
                       "Try 10 shots, then 10,000: more shots, closer to the truth.")
    st.markdown("#### Check your understanding")
    quiz("l1", "q1", "You measure the same qubit 10 times and get 7 ones. Its true chance of 1 is 60%. What does that tell you?",
         ["The chance must actually be 70%.", "Small numbers of measurements are noisy; more shots would give a closer estimate.",
          "The qubit is broken."], 1,
         "Few measurements give noisy estimates. Real quantum computers only give shots, which is why shot counts matter.")

# =========================================================================== #
# L2 CLEAN DATA
# =========================================================================== #
with tabs[1]:
    lesson_header(1, "l2")
    st.markdown("Models learn whatever is in the table, mistakes included. Duplicates inflate scores, ID columns leak identity, and "
                "typos like **240 hours studied** distort everything. Here is a small, deliberately messy table about student exam results.")
    raw = lesson_messy_table()
    st.dataframe(raw.head(12), width="stretch", height=250)
    rep0 = profile_dataframe(raw, "passed")
    st.markdown("#### Your mission: raise the data quality score to 90 or more")
    a, b = st.columns([1, 1.2], gap="large")
    with a:
        t_num = st.checkbox("Convert 'attendance' from text ('85%') to numbers", False, key="l2_num")
        t_dup = st.checkbox("Remove duplicate rows", False, key="l2_dup")
        t_id = st.checkbox("Drop the ID-like column (student_id)", False, key="l2_id")
        t_const = st.checkbox("Drop the constant column (school_year)", False, key="l2_const")
        t_fill = st.checkbox("Fill missing numbers with the median", False, key="l2_fill")
        t_clip = st.checkbox("Clip extreme values (like 240 hours)", False, key="l2_clip")
    cfg = CleaningConfig(parse_numeric_text=t_num, drop_duplicates=t_dup, drop_id_like=t_id, drop_constant=t_const,
                         standardize_text=True, drop_sparse_cols_pct=1.0, drop_sparse_rows_pct=1.0)
    cleaned, _log = clean_structural(raw, "passed", cfg)
    if t_fill or t_clip:
        feats = cleaned.drop(columns=["passed"])
        fc = FittedCleaner(CleaningConfig(impute_numeric="median", outliers="clip" if t_clip else "none", outlier_iqr=1.5)).fit(feats)
        out = fc.transform(feats) if t_fill else feats.copy()
        if t_clip and not t_fill:
            for c, (lo_, hi_) in fc.bounds.items():
                out[c] = out[c].clip(lo_, hi_)
        cleaned = out.assign(passed=cleaned["passed"].values)
    rep1 = profile_dataframe(cleaned, "passed")
    with b:
        r0, r1 = st.columns(2)
        r0.markdown(ui.card("Before", ui.score_ring(rep0.score), foot=f"{rep0.n_rows} rows · {rep0.missing_cells} gaps · {rep0.duplicate_rows} duplicates"), unsafe_allow_html=True)
        r1.markdown(ui.card("Now", ui.score_ring(rep1.score), tone="t", foot=f"{rep1.n_rows} rows · {rep1.missing_cells} gaps · {rep1.duplicate_rows} duplicates"), unsafe_allow_html=True)
    if rep1.score >= 90:
        ui.callout("good", f"<b>Mission complete.</b> Quality score {rep1.score}. The table is ready to model.")
    elif rep1.issues:
        ui.md("".join(f'<div class="q-issue"><div><span class="q-sev {i.severity}">{i.severity}</span><b>{ui.esc(i.title)}</b>'
                      f'<small>{ui.esc(i.detail)}</small></div></div>' for i in rep1.issues[:4]))
    st.markdown("#### Why the order matters")
    ui.callout("", "<b>Some steps look only at rows and columns</b> (duplicates, ID columns, text numbers). They are safe to do on all data.<br>"
                   "<b>Some steps learn a number from the data</b> (the median used to fill gaps, the limit for extremes). "
                   "Those must be learned from the <i>training rows only</i> and then applied to the test rows. "
                   "(In this small lesson we apply them to everything, to keep it simple.)")
    st.markdown("#### Check your understanding")
    quiz("l2", "q1", "Why should the median used to fill gaps be learned from the training rows only?",
         ["It is faster to compute.", "Using the test rows would let information from the test set leak into training and inflate the score.",
          "The test set has no gaps."], 1,
         "That is data leakage. The Research Workbench always fits these steps on training rows only.")

# =========================================================================== #
# L3 NUMBERS TO QUBITS
# =========================================================================== #
with tabs[2]:
    lesson_header(2, "l3")
    st.markdown("A qubit can only take a rotation angle, so every number must become an angle first. The workspace **squeezes each column into 0 to π** "
                "(about 3.14, half a turn) and then rotates one qubit by that angle. This is called **angle encoding**.")
    c1, c2 = st.columns([1, 1.4], gap="large")
    with c1:
        lo, hi = 0.0, 10.0
        val = st.slider("Hours studied (the original number)", lo, hi, 6.5, 0.5, key="l3_val")
        ang = (val - lo) / (hi - lo) * np.pi
        p1 = float(np.sin(ang / 2) ** 2)
        ui.md(ui.kpi("Angle", f"{ang:.2f} rad", f"= ({val:g} − {lo:g}) ÷ ({hi:g} − {lo:g}) × π") +
              "<div style='height:10px'></div>" + ui.kpi("Chance the qubit reads 1", f"{p1:.0%}", "= sin²(angle ÷ 2)", "mute"))
    with c2:
        t = np.linspace(0, 2 * np.pi, 200)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=np.cos(t), y=np.sin(t), mode="lines", line=dict(color="rgba(167,139,250,.5)", width=2), showlegend=False))
        fig.add_trace(go.Scatter(x=[0, np.sin(ang)], y=[0, np.cos(ang)], mode="lines+markers", line=dict(color=ui.TEAL, width=5),
                                 marker=dict(size=[4, 14], color=ui.TEAL), showlegend=False))
        fig.add_annotation(x=0, y=1.18, text="|0⟩ (reads 0)", showarrow=False, font=dict(color="#fff"))
        fig.add_annotation(x=0, y=-1.18, text="|1⟩ (reads 1)", showarrow=False, font=dict(color="#fff"))
        fig.update_xaxes(visible=False, range=[-1.5, 1.5], scaleanchor="y")
        fig.update_yaxes(visible=False, range=[-1.4, 1.4])
        st.plotly_chart(ui.style_fig(fig, 330, "The qubit's state as an arrow", legend=False), width="stretch")
    st.markdown("#### Three ways to encode")
    explain_cards([
        ("Angle encoding", "One number rotates one qubit. Simple, fast, one qubit per column. Used by default.", "sliders", ""),
        ("Amplitude encoding", "The numbers become the strengths of the whole quantum state. Packs many numbers into few qubits, but absolute size is lost.", "layers", "t"),
        ("IQP encoding", "Rotate each qubit, then link them using products of the numbers. IQP = Instantaneous Quantum Polynomial.", "zap", "a"),
    ])
    st.markdown("#### Check your understanding")
    quiz("l3", "q1", "A column has been squeezed to the range 0 to π. A row's value maps to exactly π. What does its qubit read?",
         ["Always 0", "Always 1", "0 or 1 with equal chance"], 1,
         "At angle π the qubit points to |1⟩: sin²(π ÷ 2) = 1, so it always reads 1.")

# =========================================================================== #
# L4 THE CIRCUIT
# =========================================================================== #
with tabs[3]:
    lesson_header(3, "l4")
    st.markdown("A circuit has three parts: an **encoder** (loads your numbers), **learning layers** (rotations with adjustable angles, plus links "
                "between qubits), and **measurement**. Training changes only the adjustable angles.")
    c1, c2, c3 = st.columns(3)
    nq = c1.slider("Qubits", 2, 5, 3, key="l4_q")
    nl = c2.slider("Layers", 1, 3, 2, key="l4_l")
    ans = c3.selectbox("Circuit type", ["basic_entangler", "strongly_entangling", "hardware_efficient"], index=2, key="l4_a",
                       format_func=lambda a: {"basic_entangler": "Basic entangler", "strongly_entangling": "Strongly entangling", "hardware_efficient": "Hardware efficient"}[a])
    l, r = st.columns([1.5, 1], gap="large")
    l.image(circuit_png(nq, nl, "angle", ans), caption="Encoder · learning layers · measurement", width="stretch")
    with r:
        ui.md(ui.card("Gate cheat sheet", "<ul><li><b>Rotation (RX, RY, RZ, Rot)</b>: tilts a qubit by an adjustable angle.</li>"
                      "<li><b>CNOT</b> (Controlled NOT): flips one qubit if another is 1. It links qubits, which is called entanglement.</li>"
                      "<li><b>Meter</b>: measures the qubit.</li></ul>", ic="cpu"))
    st.markdown("#### Predict, then reveal")
    guess = st.number_input("How many adjustable angles does this circuit have?", 0, 200, 0, key="l4_guess")
    per = {"basic_entangler": 1, "strongly_entangling": 3, "hardware_efficient": 2}[ans]
    if st.button("Reveal the answer", key="l4_reveal"):
        st.session_state["l4_shown"] = True
    if st.session_state.get("l4_shown"):
        total = n_circuit_params(nq, nl, ans)
        ok = int(guess) == total
        ui.callout("good" if ok else "warn", f"<b>{total} angles</b> = {nl} layers × {nq} qubits × {per} angles per qubit. "
                                              + ("You got it." if ok else f"You guessed {int(guess)}."))
    st.markdown("#### Check your understanding")
    quiz("l4", "q1", "You double the number of layers. What happens to the number of adjustable angles?",
         ["It stays the same.", "It doubles.", "It squares."], 1,
         "Angles = layers × qubits × angles per qubit, so doubling the layers doubles the angles. More angles means more to train, not always better results.")

# =========================================================================== #
# L5 TRAINING
# =========================================================================== #
with tabs[4]:
    lesson_header(4, "l5")
    st.markdown("Training repeats one loop: run the circuit, measure how wrong the answer is (the **loss**), nudge the angles to reduce it. "
                "The **learning rate** is the size of each nudge. Let's train on the Iris flower dataset with three learning rates.")
    lrs = st.multiselect("Learning rates to compare", [0.005, 0.02, 0.1, 0.5, 1.5], default=[0.005, 0.1, 1.5], key="l5_lrs")
    ep = st.slider("Epochs (passes over the data)", 5, 30, 20, key="l5_ep")
    if st.button("Train and compare", type="primary", key="l5_go") and lrs:
        with st.spinner("Training a few quantum models on a simulator…"):
            st.session_state["l5_res"] = lr_experiment(tuple(sorted(lrs)), ep)
    out = st.session_state.get("l5_res")
    if out:
        fig = go.Figure()
        for (lr, d), colour in zip(out.items(), [ui.AMBER, ui.TEAL, ui.ROSE, ui.VIOLET, ui.SKY]):
            fig.add_trace(go.Scatter(x=list(range(1, len(d["loss"]) + 1)), y=d["loss"], mode="lines", name=f"learning rate {lr}", line=dict(color=colour, width=3)))
        fig.update_xaxes(title="epoch")
        fig.update_yaxes(title="loss (lower is better)")
        st.plotly_chart(ui.style_fig(fig, 360, "How fast and how smoothly the loss falls"), width="stretch")
        tbl = pd.DataFrame({"Learning rate": list(out), "Final loss": [round(d["loss"][-1], 3) for d in out.values()],
                            "Held-out accuracy": [f"{d['acc']:.0%}" for d in out.values()]})
        st.dataframe(tbl, hide_index=True, width="stretch")
        ui.callout("", "<b>What to look for.</b> A very small rate moves slowly. A very large rate overshoots and bounces. A middle rate falls smoothly. "
                       "The best value depends on the data, which is why the workbench lets you change it and compare.")
    st.markdown("#### Check your understanding")
    quiz("l5", "q1", "The loss curve zig-zags up and down and never settles. What is the most likely cause?",
         ["The learning rate is too large.", "The dataset is too clean.", "There are too few qubits."], 0,
         "Steps that are too big overshoot the best angles, so the loss bounces. Try a smaller learning rate.")

# =========================================================================== #
# L6 FAIR COMPARISON
# =========================================================================== #
with tabs[5]:
    lesson_header(5, "l6")
    st.markdown("Is the quantum model better than a normal one? The only honest way to find out: **same data, same split, several runs, "
                "and look at the spread.** Make a prediction, then test it on the Wine dataset (178 wines, 13 chemical measurements).")
    guess = st.radio("Which model do you expect to score highest?", ["Quantum neural network", "Logistic regression", "Random forest", "They will be about equal"],
                     index=None, horizontal=True, key="l6_guess")
    runs = st.select_slider("Number of runs (different random splits)", [1, 3, 5], value=3, key="l6_runs")
    if st.button("Run the comparison", type="primary", key="l6_go"):
        with st.spinner("Training the quantum model and two classical models…"):
            st.session_state["l6_res"] = compare_lab("wine", runs, 20)
    out = st.session_state.get("l6_res")
    if out:
        summ, per, n_val, v, text, pm = out
        rows = []
        vmax = max(summ[f"{pm}_mean"].max(), 1e-9)
        for _, r in summ.iterrows():
            me = r["model"] == QNN_NAME
            rows.append(f'<tr class="{"me" if me else ""}"><td>{ui.esc(r["model"])}</td><td>{r[f"{pm}_mean"] * 100:.1f}% ± {r[f"{pm}_std"] * 100:.1f}</td>'
                        f'<td><div class="q-bar {"" if me else "alt"}"><i style="width:{100 * r[f"{pm}_mean"] / vmax:.0f}%"></i></div></td></tr>')
        ui.md('<table class="q-table"><thead><tr><th>Model</th><th>Accuracy (mean ± spread)</th><th></th></tr></thead><tbody>' + "".join(rows) + "</tbody></table>")
        ui.callout("good" if v == "comparable" else "warn", f"<b>Verdict: {v}.</b> {text}")
        if guess:
            top = summ.loc[summ[f"{pm}_mean"].idxmax(), "model"]
            ui.callout("", f"You predicted <b>{guess}</b>. The highest average was <b>{top}</b>. "
                           "When the gap is smaller than the spread, “about equal” is the honest answer.")
        with st.expander("See every run"):
            st.dataframe(per.round(3), width="stretch")
    st.markdown("#### Check your understanding")
    quiz("l6", "q1", "Model A scores 95.6% and Model B scores 94.9% on 60 test rows, in a single run. What can you conclude?",
         ["A is better.", "Nothing reliable: the gap is smaller than the noise from a small test set and a single run.", "B is better because it is simpler."], 1,
         "With 60 rows, one row is 1.7 points. Differences below the run-to-run spread are not evidence.")

# =========================================================================== #
# L7 LIMITS
# =========================================================================== #
with tabs[6]:
    lesson_header(6, "l7")
    st.markdown("Quantum machine learning is exciting and easy to over-claim. For each scenario, decide what is wrong or what can be said.")
    scenarios = [
        ("l7a", "A student scales the whole dataset first, then splits into train and test, and reports 97% accuracy.",
         ["It is fine; scaling is harmless.", "This leaks test information into training. Fit the scaler on training rows only.", "Accuracy should be 100%."], 1,
         "Leakage makes scores optimistic. The workbench fits scaling, PCA and gap filling on the training rows only."),
        ("l7b", "A paper says its quantum model “beats classical” with 96% vs 95% on one split of 40 test rows.",
         ["Strong evidence of quantum advantage.", "Weak evidence: a 1-point gap on 40 rows from one split is inside the noise.", "Proof that the circuit has more qubits."], 1,
         "One row is 2.5 points here. Use several runs, a spread and a confidence interval before claiming anything."),
        ("l7c", "A demo says “trained on a quantum computer”. The code uses a simulator on a laptop. How should it be described?",
         ["It is fine to say quantum computer.", "It should say “simulated quantum circuit”, since no quantum hardware was used.", "Simulators are the same as hardware."], 1,
         "Real devices add noise and limits (NISQ). A simulator is ideal and must be described as a simulator."),
    ]
    for sid, q, opts, ans_i, why in scenarios:
        ui.md(ui.card("Scenario", f"<p>{q}</p>", ic="alert", tone="a"))
        quiz("l7", sid, "What is the right reading?", opts, ans_i, why, mark=False)
        st.write("")
    all_ok = all(st.session_state.get(f"quiz_l7_{s[0]}_res", ("", ""))[0] == "ok" for s in scenarios)
    if all_ok and "l7" not in DONE:
        DONE.add("l7")
        st.rerun()
    if len(DONE) == len(LESSONS):
        ui.md(ui.card("You finished the Learning Lab", "<ul>" + "".join(f"<li>{t}</li>" for t in TAKEAWAYS) + "</ul>", ic="award", tone="t",
                      foot="Next step: take your own data to the Research Workbench and run a real study."))
        if not st.session_state.get("lab_balloons"):
            st.balloons()
            st.session_state["lab_balloons"] = True
        st.page_link("views/research.py", label="Open the Research Workbench", icon=":material/science:")
    else:
        ui.callout("", "Answer all three scenarios correctly to finish this lesson.")

# =========================================================================== #
# GLOSSARY
# =========================================================================== #
with tabs[7]:
    ui.section("Reference", "Glossary", "Every technical word used in the workspace, with its full form and a plain-English meaning.")
    q = st.text_input("Search terms", placeholder="Type a word, for example PCA", label_visibility="collapsed")
    items = [g for g in GLOSSARY if q.lower() in (g[0] + g[1] + g[2]).lower()]
    ui.grid([ui.card(f"{t}" + (f' <span style="color:#9a96c0;font-weight:500;font-size:.8em">· {f}</span>' if f else ""), f"<p>{m}</p>") for t, f, m in items], cols=3) if items else st.caption("No match.")

ui.footer()
