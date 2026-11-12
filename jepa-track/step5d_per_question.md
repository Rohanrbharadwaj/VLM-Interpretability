# Step 5d — Per-Question Analysis of Attribution-Patching Data

**Data:** `atp_results.npz`, `atp_components.csv` (600 questions, noise mode, $k=0.5$; 1056 components = 32×32 heads + 32 MLPs; last position).
**Conventions:** $E = \Delta_{\text{clean}} - \Delta_{\text{corr}}$ is the total image effect, where $\Delta = \text{logit(Yes)} - \text{logit(No)}$. "Raw" effects are $+$ = toward Yes. Direct effects are clean-minus-corrupted DLA, and $\sum_c \text{direct}_c = E$ (verified). "Present" and "absent" refer to the object in the image.

---

## 1. Headline

**L18.mlp is the swing component.** It carries the largest image-dependent signal (44% of $\mathrm{Var}(E)$) and it is the component whose level separates right from wrong answers. The large L20/L28 offsets are a separate gain axis that does not carry truth.

---

## 2. Which components discriminate present vs absent

AUROC of the raw direct effect for present vs absent:

| Component | AUROC | Mean raw (present) | Mean raw (absent) |
|---|---|---|---|
| L23.mlp | 0.745 | 0.327 | 0.133 |
| L16H18 | 0.744 | −0.069 | −0.092 |
| L18.mlp | 0.743 | −0.486 | −0.847 |
| L24.mlp | 0.265 | 0.111 | 0.189 |
| L16.mlp | 0.713 | −0.207 | −0.279 |
| L21.mlp | 0.336 | −0.640 | −0.532 |
| L20.mlp | 0.638 | 2.288 | 2.103 |

Group sums:

| Group | AUROC (truth) | corr with $\Delta_{\text{clean}}$ | Mean raw (present / absent) |
|---|---|---|---|
| Evidence MLPs {16,17,18,20,23} | **0.780** | +0.851 | +2.52 / +1.67 |
| Opposing MLPs {21,24,27} | 0.284 | −0.587 | −1.04 / −0.74 |
| Late MLPs {28,30,31} | 0.460 | −0.171 | −0.70 / −0.62 |
| All MLPs | 0.676 | +0.528 | |
| All heads | 0.526 | +0.127 | |

Reference AUROCs: clean $\Delta$ 0.799, $E$ 0.728, corrupted $\Delta$ 0.524.

- The evidence-MLP sum (0.780) discriminates **better than the total image effect $E$ (0.728)**: the rest of the network dilutes it.
- Opposing MLPs have AUROC $<0.5$, i.e. they respond in the opposite direction. This is consistent with a compensating role.
- The corrupted $\Delta$ (question-only prior) has AUROC 0.524, so it carries almost no truth signal.

---

## 3. Variance of the image effect

Share of $\mathrm{Var}(E)$ by component, $\mathrm{Cov}(x_c, E)/\mathrm{Var}(E)$:

| Component | Share |
|---|---|
| L18.mlp | **0.437** |
| L23.mlp | 0.167 |
| L27.mlp | −0.080 |
| L26H3 | 0.077 |
| L30H7 | 0.070 |
| L24.mlp | −0.060 |
| L16.mlp | 0.059 |
| L17.mlp | 0.054 |
| L27H7 | 0.053 |
| L20.mlp | 0.048 |

The shares sum to 1 by construction. MLPs account for 0.693 and heads for 0.307.

**Heads are diffuse:** the top 10 heads cover 23% of $\sum|\text{truth part}|$, the top 50 cover 46%, and the top 100 cover 60%. The largest are L16H0 (+0.0199), L23H10 (−0.0168), L16H18 (+0.0119), and all are small next to L18.mlp.

---

## 4. Evidence and opposing MLPs are coupled per question

Within-class correlations of raw direct effects (group sums):

| Class | corr(EV, OP) | corr(EV, BR) | corr(OP, BR) | corr(EV, $\Delta_{\text{clean}}$) |
|---|---|---|---|---|
| Absent | **−0.76** | −0.51 | +0.54 | +0.80 |
| Present | **−0.68** | −0.16 | +0.32 | +0.81 |

Here EV = {16,17,18,20,23}, OP = {21,24,27}, BR = {28,30,31}.

Selected pairs (all questions): corr(L20, L21) = −0.63, corr(L23, L24) = −0.46, corr(L20, L28) = −0.40, corr(L18, L20) = 0.01, corr(L18, L28) = 0.05.

**Reading:** when the evidence MLPs push harder, the opposing MLPs push back harder on the same question. This fits compensation or self-repair, but also fits both groups reading the same residual. These correlations cannot separate the two; only ablation and patching can. L18 is statistically independent of L20 and L28.

---

## 5. A separate "gain" axis (PCA of the 32 MLP effects)

Variance explained by PC1–PC5: 0.425, 0.203, 0.141, 0.054, 0.038.

PC1 loadings: L28 (+0.83), L20 (−0.39), L27 (+0.29), L23 (−0.15), L29 (−0.14), L21 (+0.09), L17 (−0.07), L31 (−0.06).

- PC1 vs truth: AUROC 0.597. Corr with $\Delta_{\text{clean}}$ is −0.30 and corr with $E$ is −0.14.
- PC1 is a L28-vs-L20 axis that varies per question but is not evidence. The large constants (L20 ≈ +2.4, L28 ≈ −1.9) belong to this axis, so part of the aggregate "signal" in the 5d numbers is this gain.

---

## 6. What distinguishes errors

Mean raw direct effects by outcome:

| Outcome | L18.mlp (sd) | L20.mlp | L28.mlp | Mean $\Delta_{\text{clean}}$ |
|---|---|---|---|---|
| TP | −0.30 (0.33) | 2.38 | −1.27 | +0.49 |
| FN | −0.74 (0.31) | 2.16 | −1.31 | −0.33 |
| TN | −0.95 (0.35) | 2.08 | −1.18 | −0.64 |
| FP | −0.45 (0.28) | 2.19 | −1.35 | +0.23 |

- Within each class, right and wrong answers differ in L18.mlp by ~0.4–0.5, which is more than 1 sd. L20 and L28 barely move across outcomes.
- L18.mlp is **negative in every class**: "object present" appears as *less suppression* of Yes, not as a push toward Yes.

**Linear what-if (direct effect only).** For erroneous questions, replace one component's raw effect with its mean on correctly-answered questions of the same class:

| Component | FN → flip to Yes | FP → flip to No |
|---|---|---|
| L18.mlp | **0.73** | **0.83** |
| L20.mlp | 0.35 | 0.37 |
| L28.mlp | 0.35 | 0.25 |
| L31.mlp | 0.20 | 0.18 |
| L23.mlp | 0.30 | 0.50 |
| L26H3 | 0.23 | 0.52 |

This is a screening tool and an **upper bound**: it ignores downstream compensation (see §4), and "correct" is defined by the same $\Delta$, so it is partly circular.

---

## 7. Correctness vs image effect and prior

Accuracy by quintile (low to high):

| Variable | Q1 | Q2 | Q3 | Q4 | Q5 |
|---|---|---|---|---|---|
| Signed $E$ (toward correct) | 0.31 | 0.58 | 0.73 | 0.87 | 0.98 |
| Corrupted $\Delta$ (question-only prior) | 0.65 | 0.73 | 0.66 | 0.71 | 0.72 |

The prior does not predict correctness, whereas the image effect does. Because $\Delta_{\text{clean}} = \Delta_{\text{corr}} + E$, the first row is partly circular and is descriptive only.

---

## 8. Caveats

- First-order, **direct** effects at the last position only. Indirect effects are excluded.
- Noise corruption at $k=0.5$. The corrupted baseline has AUROC 0.524, so "effect" here largely means "effect relative to a noisy image", not relative to a no-object baseline.
- `none` and `shuffle` modes have not been run, so L18's role is unconfirmed outside noise mode.
- Not reconciled: $\sum$ constants ($-0.224$) vs mean raw $E$ ($-0.342$). Do not quote either as exact until this is resolved.
- 600 questions, one model run; no confidence intervals on the correlations above yet.

---

## 9. Predictions for steps 6–7

1. Mean-ablating L18.mlp lowers AUROC more than any other single component.
2. The evidence group {16,17,18,20,23} lowers AUROC more than a random MLP set of the same size.
3. L20.mlp and L28.mlp ablations shift the yes-rate (bias) but leave AUROC nearly unchanged.
4. The opposing group {21,24,27} ablation leaves AUROC unchanged or slightly higher.
5. Patching the layer 16–19 window flips answers more than neighbouring windows.
6. Because of the compensation in §4, the realised effects will be smaller than the §6 what-if.
