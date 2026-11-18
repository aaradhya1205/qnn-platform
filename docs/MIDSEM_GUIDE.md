# Mid-term evaluation guide: Quantum Image Lab

Everything you need to present, explain and defend the project in a 20-minute interaction.
Every number below was measured with this code (see "Numbers to remember"); re-check them
on your own machine before the session because results shift by a few points between seeds.

---

## 0. The project in 30 seconds (every group member should know this by heart)

> We classify small grayscale images with a **quantum neural network**. An image is first
> reduced to a tiny grid (4×4), then **written into the state of 5 qubits** using the FRQI
> quantum image representation (we also compare QPIE amplitude encoding and angle encoding).
> A **trainable circuit** of rotation and CNOT gates processes that state, we **measure each
> qubit**, and a small linear layer turns the measurements into a class. We run it on a
> simulator, train it with gradient descent, and compare it against logistic regression, SVM,
> a small neural network and a random forest on exactly the same images. We also run a second
> quantum image-processing task, **quantum edge detection**, which needs no training.
> The honest finding: the quantum model reaches about **98%** like the classical ones
> (~99%), so there is **no quantum advantage here**, and we can explain exactly why and what
> the limits are.

**Do not claim** a quantum speed-up or better accuracy. The rubric rewards understanding
of "where quantum helps, where it does not". Saying so confidently is the strongest answer.

---

## 1. Launch checklist (do this tonight, again 30 minutes before)

```bash
cd ~/Desktop/Projects/qnn-platform
source .venv/bin/activate
streamlit run app.py          # opens in your browser (default http://localhost:8501)
pytest -q                     # 67 tests should pass (about 10 s)
```

1. Open the app. In the sidebar press **🚀 Run pipeline** once (about 10 s). Do not refresh the
   browser tab afterwards: a refresh clears the trained model and experiment results.
2. Open tab **8 Experiments**, tick **Rigorous (3 seeds)**, and run experiments **A, B, C, D, E, F,
   G, H** so the charts are ready. Total about 4 minutes. Keep the tab open.
3. Have two or three images ready to upload from `demo_images/` (for example `digit_3.png`,
   `digit_8_rotated_noisy.png`, and `digit_5.png` as a deliberate failure case).
4. Take screenshots of tabs 3, 4, 7, 8 as a backup in case something misbehaves.
5. Close other apps, plug in the charger, set the display to never sleep. No internet is required.

If anything breaks: `Ctrl+C` in the terminal, run `streamlit run app.py` again, press Run pipeline.

---

## 2. 20-minute demo script (what to click, what to say)

| Time | Tab | Do | Say |
|---|---|---|---|
| 0:00 | Overview | Show the problem, the pipeline diagram and the criteria table | Problem, why quantum, research question (section 3) |
| 2:00 | 1 Input | Upload `demo_images/digit_3.png` (or take a photo) | "This is a real image file, not from the training set. We will follow it through every stage." |
| 3:00 | 2 Preprocess | Point at the six panels left to right | Grayscale, crop, invert, resize to 4×4, normalise. "Everything the quantum circuit ever sees is these 16 numbers." |
| 4:30 | 3 Encode | Show formula, table, probability bars, the green check | FRQI explained in section 5. "The circuit-built state equals the textbook formula to 10⁻¹⁶." |
| 7:00 | 4 Circuit | Toggle "Expand the FRQI block" | Read the circuit in 30 seconds (section 6). Point at the gate count: 1,248 gates |
| 9:00 | 5 Train | Show loss and accuracy curves | Backprop through the simulator; parameter-shift on hardware |
| 10:00 | 6 Measure | Show ⟨Z⟩ bars and prediction; move the shots slider to 10 | "On real hardware you only get shots. With 10 shots the prediction is unstable." |
| 11:30 | 7 Evaluate | Table, confidence interval, confusion matrices, mistakes | Same split, same pixels. "Differences below the confidence interval are noise." |
| 13:00 | 8 Experiments | Show A, B, C, E, F, H results (already computed) | One finding per chart (section 7) |
| 15:30 | 9 Advantage & limits | Read the two cards and the limitations | Section 8 |
| 17:00 | ⭐ Bonus: edges | Same image, 16×16 | "A second quantum image algorithm. One Hadamard gives edges. Error vs classical: 10⁻¹⁶." |
| 18:30 | Q&A | | Section 10 |

The evaluators said this is **not only a demo**. Spend more time on *why* (formulation, experimental
design, limitations) than on clicking.

---

## 3. Evaluation criteria: what to say and where to show it

### 3.1 Problem and objective
- **Problem:** classify (and process) images using a quantum circuit when a quantum device
  can only hold a handful of qubits, while images have many pixels.
- **Research question:** *How can an image be represented as a quantum state and processed by a
  trainable circuit, how many resources does that take, and does it compete with classical models?*
- **Motivation for a quantum approach (be honest):**
  1. N pixels fit in about log₂N qubits (a state has 2ⁿ amplitudes), a very compact representation.
  2. Some image operations become constant-depth circuits (edge detection with one Hadamard).
  3. It is an open question whether such models can learn as well with far fewer parameters, and this
     project measures it. Our QNN has **26 trainable parameters** versus **1,089** for the small classical neural network.
- **Show:** Overview tab.

### 3.2 Quantum formulation
- **FRQI** (Flexible Representation of Quantum Images): state
  `|I⟩ = 1/2ⁿ Σᵢ (cos θᵢ |0⟩ + sin θᵢ |1⟩) ⊗ |i⟩`, with `θᵢ = pᵢ·π/2`.
  The first 2n qubits store *which pixel*, the last qubit stores *how bright it is*.
- **QPIE:** the normalised pixel vector becomes the amplitude vector: `|I⟩ = Σ pᵢ/‖p‖ |i⟩`.
- **Angle + PCA:** compress to k numbers, one rotation `RY(φⱼ)` per qubit.
- **Algorithm for a second task:** quantum Hadamard edge detection (QHED).
- **Connection to the image task:** pixel intensity → rotation angle or amplitude; the trainable
  circuit mixes pixel information through entanglement; measurement gives features for classification.
- **Show:** tab 3 (green check proves the circuit state equals the formula), tab 4, bonus tab.

### 3.3 Methodology and implementation
- **Framework:** PennyLane (circuits, simulator, differentiation) + PyTorch (training) + scikit-learn (baselines, PCA).
- **Preprocessing (tab 2):** grayscale → centre crop → optional invert → area-average resize to N×N → per-image min-max to [0,1].
- **Circuit (tab 4):** encoder → L layers of (single-qubit rotations + CNOT chain/ring) → measure ⟨Z⟩ on every qubit → linear layer.
- **Training (tab 5):** Adam, learning rate 0.08, batch 16, 12 epochs; binary cross-entropy (2 classes) or cross-entropy.
- **Simulation:** exact state-vector simulator (`default.qubit`), plus finite-shot sampling and a depolarising-noise
  density-matrix simulator (`default.mixed`) for the non-ideal experiments. **No real quantum hardware was used.**
- **Show:** tabs 2, 4, 5. Code: `qnn_platform/quantum_image.py`, `image_pipeline.py`, `image_data.py`.

### 3.4 Experimental evaluation
- **Dataset:** scikit-learn handwritten digits (1,797 images of 8×8 pixels), default task *3 versus 8*
  (visually similar). 120 images per class, stratified 75/25 train/test split. Any ZIP of class folders can also be uploaded.
- **Metrics:** accuracy, precision, recall, F1, confusion matrix, 95% Wilson confidence interval, training time, parameter count.
- **Baselines:** logistic regression, SVM (RBF), MLP (32-16 hidden units), random forest, on the same pixels and the same split.
- **Rigour:** multi-seed mode (mean ± std), confidence interval on the small test set, ablation of the circuit.
- **Show:** tab 7 and tab 8.

### 3.5 Results and comparison
Quote section 9 ("Numbers to remember"). Interpretation in one sentence:
*"All five models land within about one point of each other; with 60 test images that difference is inside the noise, so the correct statement is that the quantum model is competitive, not better."*

### 3.6 Quantum advantage and analysis
- **Where it helps:** qubit count (log of pixels), parameter count (26 vs 1,089), constant-depth image operations.
- **Where it does not:** accuracy (no gain), loading cost (FRQI 4×4 compiles to 1,248 gates, 8×8 to 16,558),
  readout (needs many shots), simulation cost (doubles per added qubit).
- **Ablation:** with a *random frozen* circuit the accuracy drops from 98.3% to 81.1%, so training the circuit matters.
  But classical logistic regression still reaches 98.9% with 17 parameters.
- **Show:** tab 9 and experiment H.

### 3.7 Limitations and failure cases
Tab 9 has the full list. Memorise these five:
1. Tiny images (4×4) on an easy two-class problem; accuracy does not extrapolate to real images.
2. Simulator only; real hardware adds gate errors, decoherence, limited connectivity.
3. State preparation cost grows with pixel count (FRQI) and readout needs many shots.
4. Small test set (60 images, 95% confidence interval roughly ±5-6 points).
5. Domain shift and unseen classes: a photo with different lighting, or a digit the model was never trained on, is misclassified (try `digit_5.png`).

### 3.8 Individual understanding
Every member should be able to answer the questions in section 10 and give the 30-second summary in section 0.
Suggested split for *presenting* (everyone still studies everything): one person Overview + formulation (tabs 1-3),
one the circuit and training (4-5), one results and experiments (6-8), one advantage, limitations and edge detection (9, bonus).

---

## 4. Parameter cheat sheet (what each sidebar control does and what to set)

| Parameter | What it means | Effect when increased | Demo value |
|---|---|---|---|
| **Digit classes** | Which classes to separate | More classes = harder problem, more outputs | 3 and 8 |
| **Max images per class** | Data per class | More data = more stable accuracy, slower training | 120 |
| **Representation** | How pixels become a quantum state | FRQI: 2n+1 qubits, many gates. QPIE: 2n qubits, fewer gates. Angle+PCA: k qubits, lossy | FRQI (then compare) |
| **Image size** (N per side) | Resolution fed to the circuit | Qubits grow like 2·log₂N+1, **gates and simulation time grow much faster** | 4 (7+ qubits is slow) |
| **PCA features** (angle only) | Number of qubits | More features = more information, more qubits | 4 |
| **Ansatz** | Pattern of trainable gates | Strongly entangling: most expressive, most parameters. Basic: fewest | Hardware-efficient |
| **Layers** (depth) | Repetitions of the trainable block | More parameters and gates; accuracy plateaus after 2-3; risk of barren plateaus | 2 |
| **Epochs** | Passes over the training data | More = better fit, risk of overfitting, slower | 12 |
| **Batch size** | Images per gradient step | Small = noisy but frequent updates | 16 |
| **Optimizer** | Update rule | Adam is the safe default | adam |
| **Learning rate** | Step size | Too high = unstable loss, too low = slow | 0.08 |
| **Test split** | Fraction held out | Larger = more reliable accuracy estimate, less training data | 0.25 |
| **Random seed** | Controls split and initial angles | Different seeds can move accuracy by a few points | 0 |

**Rules of thumb to say out loud**
- *Qubits* = (position qubits) + (colour qubit). For a 2ⁿ×2ⁿ image: 2n + 1.
- *Circuit parameters* = layers × qubits × (rotations per qubit). Hardware-efficient has 2 per qubit per layer: 2 layers × 5 qubits × 2 = **20**.
- *Total parameters* = circuit parameters + head (qubits × outputs + outputs) = 20 + 6 = **26**.
- *Shots*: statistical error shrinks like 1/√shots; 100 shots is about 10% error per expectation value, 10,000 shots about 1%.

**How to choose when asked "why these values?"**
Start from a data/size budget (4×4 = 5 qubits trains in seconds), pick a middle ansatz and 2 layers,
and then *justify by experiment*: depth sweep B shows 2-3 layers is enough, size sweep C shows 2×2 is too small,
data sweep D shows accuracy stops improving beyond about half the data.

---

## 5. Explaining the encodings in plain words

**FRQI.** Imagine a 4×4 image. Label its 16 pixels with 4-bit addresses (0000 … 1111). Four qubits
hold an address; one extra qubit holds the brightness. The state is a *superposition over all
addresses at once*, and attached to each address is a single qubit tilted by an angle proportional to
that pixel's brightness (dark = |0⟩, bright = |1⟩). To build it: put the four address qubits in
equal superposition with Hadamard gates, then for each pixel apply a rotation to the brightness qubit
*only when the address qubits equal that pixel's address* (a multi-controlled RY gate).
Measuring the brightness qubit, given the address, returns 1 with probability sin²θ, which is how
we read the pixel back (tab 3, "read the image back by measurement").

**QPIE.** Treat the 16 brightness values as the 16 amplitudes of a 4-qubit state (after dividing by
their length so the state is normalised). Cheaper to build, but absolute brightness is lost.

**Angle + PCA.** Compress the image to a few numbers and write each one as a rotation of its own qubit.
Fewest qubits and gates, but the compression is learned from data and is lossy.

**Why three?** To show the trade-off: the faithful, invertible representation (FRQI) costs 1,248 gates;
the cheap ones (QPIE 40, angle 26) classify just as well. That is an *evidence-based* conclusion, not an opinion.

---

## 6. How to read and explain the circuit (30-second version)

1. "Each horizontal line is a qubit. There are five, because 4 address qubits + 1 brightness qubit."
2. "The tall box on the left is the **encoder**. It loads the image. It has no trainable parameters."
   (Toggle *Expand* to show it is Hadamards plus 16 multi-controlled rotations.)
3. "Then two **layers**. In each layer every qubit gets two rotations, RY and RZ. Those angles are the
   numbers training adjusts. Then a chain of **CNOT** gates (● control, ⊕ target) links neighbouring
   qubits so information from different pixels can interact. Without CNOTs the qubits would be independent
   and the circuit could not combine pixels."
4. "At the end every qubit is **measured**. The meter symbols return the expectation of Z, a number between −1 and 1."
5. "A small linear layer, 5 inputs to 1 output, turns those five numbers into the class probability."
6. "Training changes the 20 angles and the 6 head weights to reduce the classification loss."

**If asked for more detail**
- *Why expectation of Z?* It is the simplest observable; it is the bias of each qubit towards |0⟩ vs |1⟩ and can be estimated from shots.
- *Why a classical head?* Standard hybrid design: the circuit outputs a few real numbers, and a linear map converts them into class scores. The ablation (experiment H) shows the circuit's training still matters.
- *How are gradients computed?* On the simulator by backpropagation through the state vector. On hardware by the
  **parameter-shift rule**: evaluate the circuit at θ ± π/2 and combine, costing two circuit runs per parameter per image.
- *Is it entangled?* Yes, the CNOT chain creates entanglement across the register.

---

## 7. What each experiment shows (tab 8) and the sentence to say

| Experiment | Typical result (3 seeds) | Sentence |
|---|---|---|
| **A Encodings** | FRQI 98.3% (1,248 gates), QPIE 98.3% (40), angle+PCA 97.8% (26) | "Accuracy is the same, cost differs by 50×. For classification the expensive encoding buys nothing." |
| **B Depth** | 1 layer 93.9%, 2 layers 98.3%, 3 layers 98.9%, 4 layers 96.7%, 6 layers 97.2% | "Two to three layers saturate. More depth adds parameters, not accuracy, and risks barren plateaus." |
| **C Image size** | 2×2 85.6%, 4×4 98.3% | "Too little resolution loses the class information. 8×8 (7 qubits) is possible but about ten times slower to simulate." |
| **D Data size** | 10% 63.9%, 25% 92.8%, 50% 97.8%, 100% 98.3% | "Learning curve flattens beyond half the data. The QNN is data-efficient on this task." |
| **E Shots** | exact 100%, 10 shots 80%, 100 shots 97%, 1,000 shots 99% | "On hardware you need hundreds of shots per circuit run. This is the readout cost." |
| **F Noise** | p ≤ 0.05: 97-100%, p = 0.15: 55% (chance) | "Tolerates mild noise, collapses to chance at high noise. Depolarising noise shrinks every ⟨Z⟩ towards zero." |
| **G Scaling** | FRQI: 90 → 1,248 → 16,558 gates for 4 → 16 → 64 pixels | "Qubits grow logarithmically, gates grow faster than the pixel count. State preparation is the bottleneck." |
| **H Ablation** | trained 98.3% vs random frozen circuit 81.1% | "The circuit is not just a fixed feature map, training it contributes about 17 points." |

Single runs fluctuate (for example 93-100% for the default setting). **That is why multi-seed mode and the confidence interval exist.** Use them when making any comparison.

---

## 8. Quantum advantage: the honest answer (memorise this)

- **Accuracy:** no advantage. 98.3% ± 2.4 for the QNN versus 98.9-99.4% classical, differences inside noise.
- **Parameters:** a real, if modest, point in favour: 26 versus 1,089 for the MLP. But logistic regression already reaches 98.9% with 17.
- **Qubits vs pixels:** exponential compression in *storage*: 64 pixels in 7 qubits (FRQI) or 6 (QPIE).
- **Why that is not a speed-up:** to load an image you still touch every pixel. FRQI needs at least one multi-controlled
  rotation per pixel, and compiled to a hardware basis the cost grows faster than the pixel count (16,558 gates for 8×8).
  More efficient constructions exist (for example Gray-code based, or NEQR) but still O(N) at best.
  Then reading the result needs many shots. Hence "loading in, reading out" dominates any processing gain.
- **Where a quantum circuit really is cheap:** the *processing* step itself, for example edge detection is one Hadamard.
  But the surrounding load/readout costs erase the benefit for a single image.
- **Where an advantage might appear in future:** data that is *already quantum* (outputs of quantum sensors/experiments), or
  much larger images with efficient state preparation (for example qRAM), or problems where classical models struggle.
  None of that is claimed or tested here.
- **Bottom line:** *"We built and verified the full quantum image pipeline and measured its costs. It is competitive but not better, and the experiments say exactly why."*

---

## 9. Numbers to remember (default setup: 3 vs 8, FRQI 4×4, 2 layers, 12 epochs, 120 images/class, 3 seeds)

| Quantity | Value |
|---|---|
| Qubits (FRQI 4×4 / 8×8) | 5 / 7 |
| Qubits (QPIE 4×4) | 4 |
| Trainable parameters (circuit / head / total) | 20 / 6 / 26 |
| Baseline parameters (logistic regression / MLP) | 17 / 1,089 |
| QNN test accuracy | **98.3% ± 2.4** (single runs 93-100%) |
| Logistic regression / SVM / MLP / random forest | 98.9% / 99.4% / 98.9% / 98.9% |
| Test set size / 95% confidence interval | 60 images / roughly ±5-6 points |
| Compiled gates, FRQI 2×2 / 4×4 / 8×8 | 90 / 1,248 / 16,558 |
| Compiled CNOTs, FRQI 4×4 / 8×8 | 392 / 6,156 |
| Compiled gates, QPIE 4×4 / 8×8 | 40 / 102 |
| Training time (4×4 FRQI, 12 epochs) | about 5-8 s on a laptop simulator |
| QHED error versus classical differences | about 10⁻¹⁶ (exact), 8 qubits for 16×16 |
| Frozen random circuit (ablation) | 81.1% |

---

## 10. Question bank with model answers

**Problem and motivation**
1. *Why quantum for images at all?* Compact state representation, native linear algebra on all amplitudes, some operations constant depth. We test whether that translates into learning performance, and find it does not beat classical here.
2. *Why digits and not real photos?* A quantum simulator doubles in cost per qubit; real photos need far more qubits than any simulator or current device offers. Small digits keep the full pipeline checkable end to end.
3. *Is this really "image processing"?* Yes: preprocessing, quantum image representation, classification, and a genuine quantum image algorithm (edge detection).

**Formulation**
4. *Write the FRQI state.* `|I⟩ = 2⁻ⁿ Σ (cos θᵢ|0⟩ + sin θᵢ|1⟩)|i⟩`, θᵢ = pᵢπ/2. Normalisation 2⁻ⁿ because there are 2²ⁿ terms each of norm 1/2ⁿ.
5. *How many qubits for a 256×256 image?* n = 8, so 2·8 + 1 = **17 qubits** for FRQI, but 65,536 multi-controlled rotations to load it.
6. *Difference between FRQI and QPIE?* FRQI stores intensity as an *angle* on a dedicated qubit (needs 2n+1 qubits, retains absolute brightness); QPIE stores it as an *amplitude* (2n qubits, loses global scale, cheaper to prepare in our compilation).
7. *Why must θ be in [0, π/2]?* So cos and sin are both non-negative and intensity maps monotonically to the angle.
8. *What does CNOT do?* Flips the target if the control is 1. It creates entanglement between qubits.
9. *Why does the circuit need entanglers?* Without them every qubit is processed independently and no information from different pixels can combine.

**Methodology**
10. *How is it trained?* Mini-batch Adam on cross-entropy; gradients by backpropagation through the simulated state; on hardware by parameter-shift.
11. *What are the trainable parameters?* The rotation angles (20) and the linear head (6).
12. *Why not just use a bigger circuit?* Diminishing returns (experiment B), more gates, barren plateaus, slower simulation.
13. *How do you avoid overfitting?* Held-out test set never used for training, small model, train-vs-test gap reported (tab 5 flags large gaps).
14. *What is the role of the classical head?* It maps the measured numbers to class scores. Ablation H shows the trained circuit adds about 17 points over a random one.

**Experiments and results**
15. *Is the difference between 98.3% and 99% significant?* No. With 60 test images the 95% interval is roughly ±5-6 points. Use multi-seed means and say "comparable".
16. *Why is a single run sometimes 93%?* Different random split and initial angles. Small data gives high variance; that is why we report mean ± std over seeds.
17. *Is the baseline fair?* Same split and pixels, default hyperparameters. Neither side is heavily tuned; we say so as a limitation.
18. *Why does noise at p = 0.15 give 55%?* Depolarising noise shrinks every expectation value toward 0; the classifier's decision rides on small differences, so it collapses toward chance (two balanced classes ≈ 50-55%).

**Quantum advantage and limits**
19. *Where is the quantum advantage?* Not in accuracy. In compactness of representation and, potentially, constant-depth operations. Loading and readout costs remove practical speed-up for a single image.
20. *Would real hardware do better?* No: noise, limited qubits and connectivity. Error mitigation and error correction would be needed.
21. *What happens for a digit not in the classes?* It is forced into a trained class. Shown with `digit_5.png`.
22. *Can this scale to MNIST (28×28)?* 28 is not a power of two, so pad or resize to 32×32: 11 qubits for FRQI, 1,024 rotations to load, feasible to simulate but slow; ten classes would be much harder for this small circuit.
23. *What would you do next?* Real-device or noise-model runs, multi-class, more efficient state preparation (NEQR, Gray-code), error mitigation, a convolution-like quantum layer.

**Edge detection**
24. *How does QHED work?* Amplitude-encode the image; a Hadamard on the least-significant qubit turns each adjacent pair (a, b) into (a+b)/√2 and (a−b)/√2; the probability of the "odd" states is the squared difference. A second run on the pixel index shifted by one handles the other pairs; repeat along columns.
25. *Is this faster than a classical filter?* The circuit is constant depth, but loading is O(N) and readout needs many shots, so no practical speed-up.

---

## 11. Things not to say
- "Quantum is faster / more accurate." (Not shown.)
- "It runs on a real quantum computer." (It is a simulator.)
- "98% proves it works." (60 test images, small easy task.)
- "FRQI uses only 5 gates." (It needs 1,248 compiled gates for 4×4.)
- Do not guess when asked something you are unsure of: say "we did not test that, but we would expect… and here is how we would test it."

---

## 12. Glossary
- **Qubit:** two-level quantum system; state α|0⟩ + β|1⟩ with |α|² + |β|² = 1.
- **Superposition:** a state that combines basis states with amplitudes.
- **Entanglement:** correlation between qubits that cannot be described qubit by qubit.
- **Gate:** a reversible operation on qubits (RY/RZ rotations, Hadamard, CNOT).
- **Ansatz:** the chosen pattern of trainable gates.
- **Expectation value ⟨Z⟩:** average measurement outcome (+1 for |0⟩, −1 for |1⟩) of one qubit.
- **Shots:** number of repeated circuit executions used to estimate expectation values.
- **Depolarising noise:** with probability p, replace a qubit by the maximally mixed state.
- **Parameter-shift rule:** exact gradient from two shifted circuit evaluations.
- **Barren plateau:** region where gradients vanish exponentially in qubit count/depth.
- **State preparation:** the circuit that loads classical data into a quantum state.
- **NISQ:** noisy intermediate-scale quantum (today's hardware).

---

## 13. Troubleshooting
| Problem | Fix |
|---|---|
| `streamlit: command not found` | `source .venv/bin/activate` first |
| Port already in use | `streamlit run app.py --server.port 8600` |
| Page shows "Run pipeline first" | Press 🚀 in the sidebar; do not refresh the browser afterwards |
| Training feels slow | Use 4×4 (5 qubits); 8×8 FRQI takes about 40 s |
| Camera option blank | Allow camera in the browser; or use Upload instead |
| Uploaded photo predicted wrongly | Check "invert" matches your image (dark ink on light paper = ticked), crop the digit tightly, use dark pen on white paper |
| Experiments take long | Untick Rigorous mode (1 seed) |
| Need demo images again | `python scripts/make_demo_images.py` |
| Verify everything | `pytest -q` |

Project layout: `app.py` (router), `image_dashboard.py` (this dashboard), `tabular_app.py` (earlier tabular platform),
`qnn_platform/quantum_image.py` (encodings, circuits, QHED), `image_pipeline.py` (training/evaluation),
`image_experiments.py` (sweeps), `image_data.py` (datasets and preprocessing), `tests/` (67 tests).
