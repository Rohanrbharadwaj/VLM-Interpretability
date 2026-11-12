# Step 7: patching sweeps. Results and conclusions

**Setting.** JEPA-VLM (I-JEPA ViT-H/16 → projector → Vicuna-7B-v1.5, 4-bit), noise corruption ($k=0.5$), 300 balanced POPE questions (158 present, 142 absent; first 300 of the step 4/6 evaluation set). For each window of layers the clean run is patched with the corrupted run's activations in two ways:

- **Last-position patching:** the outputs of all heads, all MLPs, or both at the final token, in the window.
- **K/V patching:** the cached keys and values of the **image tokens**, of the **text tokens** (BOS, instruction and question; called "question" below), or both, in the window.

Metric: $\text{frac}=\text{shift}/\text{full-corruption shift}$, where shift is the mean change of $\Delta=\text{logit(Yes)}-\text{logit(No)}$ toward the correct answer. 95% intervals are bootstrap intervals of the ratio (2000 resamples). For balanced classes the signed shift is essentially half of the loss in the present–absent gap, so $\text{frac}$ is approximately the **share of the gap loss that the patch reproduces**.

**Baselines.** Clean: AUROC $0.818$, accuracy $0.703$, yes-rate $0.437$, gap $0.688$. Corrupted: AUROC $0.516$, accuracy $0.520$, yes-rate $0.660$, gap $0.032$. Full-corruption signed shift: $-0.311$ [−0.397, −0.225].

---

## 1. Conclusions

1. **Image information reaches the answer through layers 8–19 and almost nothing else.** Patching all last-position heads and MLPs together in layers 8–15 reproduces $0.70$ [0.40, 0.97] of the effect and in layers 16–19 $0.71$ [0.60, 0.87]. Layers 0–7 reproduce $0.07$ [−0.25, 0.31], layers 20–23 $0.22$ [0.04, 0.49] and layers 24–27 $0.28$ [0.04, 0.62].
2. **The last token does not read the image tokens directly.** Patching the image-token keys and values reproduces nothing in any window: $-0.157$ [−0.212, −0.116] in 8–15, $+0.048$ [0.027, 0.070] in 16–19, and $\le0.03$ elsewhere.
3. **The last token reads the image information from the text-token positions.** Patching the text-token keys and values reproduces $0.527$ [0.395, 0.672] of the effect in layers 8–15 and $0.334$ [0.249, 0.418] in layers 16–19, and nothing from layer 20 on ($0.033$, $0.038$, $-0.004$). Difference text − image: $+0.684$ [0.539, 0.861] in 8–15 and $+0.286$ [0.210, 0.362] in 16–19.
4. **Both heads and MLPs at the last position carry the effect, redundantly.** In 8–15 heads reproduce $0.66$ [0.43, 0.91] and MLPs $0.56$ [0.15, 0.92]; in 16–19 heads $0.49$ [0.39, 0.62] and MLPs $0.55$ [0.45, 0.68]. Together they reproduce $0.70$ and $0.71$, far less than the sum of the parts, so they carry overlapping information.
5. **After layer ~19 the image information is already inside the last-position residual stream.** Neither last-position patches nor K/V patches in layers 20–27 reproduce more than about a quarter of the effect, and their per-question effect is negatively correlated with the total effect ($r=-0.30$ and $-0.32$), consistent with downstream compensation.
6. **The late MLPs (28–31) regulate gain and bias; they do not transmit the image effect.** Patching their outputs from the corrupted run drives the yes-rate to $0.98$ and widens the margin (sd $0.62\to1.25$). Their $\text{frac}$ of $-1.0$ is an artefact of that bias and the class imbalance, not a sign that the corruption "helps".
7. **Corruption removes evidence and adds a yes-bias, and both enter in layers 8–15.** Patching the last-position heads, MLPs or both in 8–15 sends the yes-rate to $0.90$–$0.96$ (accuracy on absent objects $0.03$–$0.13$). Patching only the text-token K/V in 8–15 reproduces the corrupted run's behaviour closely: yes-rate $0.64$ (corrupted $0.66$), gap $0.33$.

---

## 2. Results

### Last-position patching

| Layers | Heads: frac [CI] | MLPs: frac [CI] | Both: frac [CI] | Both: AUROC | Both: yes-rate |
|---|---|---|---|---|---|
| 0–7 | 0.07 [−0.25, 0.31] | −0.08 [−0.48, 0.22] | 0.07 [−0.25, 0.31] | 0.784 | 0.84 |
| 8–15 | **0.66** [0.43, 0.91] | **0.56** [0.15, 0.92] | **0.70** [0.40, 0.97] | 0.629 | 0.96 |
| 16–19 | **0.49** [0.39, 0.62] | **0.55** [0.45, 0.68] | **0.71** [0.60, 0.87] | 0.706 | 0.57 |
| 20–23 | 0.03 [−0.01, 0.08] | 0.19 [0.06, 0.39] | 0.22 [0.04, 0.49] | 0.793 | 0.18 |
| 24–27 | 0.11 [0.01, 0.24] | 0.17 [−0.03, 0.45] | 0.27 [0.04, 0.62] | 0.772 | 0.13 |
| 28–31 | −0.04 [−0.12, 0.02] | −1.00 [−2.30, −0.07] | −1.11 [−2.65, −0.05] | 0.750 | 1.00 |

### K/V patching

| Layers | Image KV: frac [CI] | Text KV: frac [CI] | Both KV: frac [CI] | Text KV: AUROC |
|---|---|---|---|---|
| 0–7 | −0.18 [−0.47, 0.03] | −0.05 [−0.28, 0.13] | 0.07 [−0.25, 0.31] | 0.790 |
| 8–15 | −0.16 [−0.21, −0.12] | **0.53** [0.40, 0.67] | **0.57** [0.45, 0.73] | 0.703 |
| 16–19 | 0.05 [0.03, 0.07] | **0.33** [0.25, 0.42] | **0.38** [0.28, 0.47] | 0.796 |
| 20–23 | 0.03 [−0.00, 0.07] | 0.03 [−0.00, 0.08] | 0.08 [0.01, 0.17] | 0.818 |
| 24–27 | −0.01 [−0.03, 0.00] | 0.04 [−0.02, 0.12] | 0.04 [−0.02, 0.11] | 0.819 |
| 28–31 | 0.02 [−0.04, 0.07] | −0.00 [−0.07, 0.05] | 0.01 [−0.09, 0.09] | 0.803 |

In window 0–7 the "both" rows equal the last-position heads row exactly. This is expected: with all heads in layers 0–7 patched at the last position, the residual stream is already the corrupted run's, so patching the MLPs on top adds nothing. The same holds for the K/V "both" case.

### Per-question agreement with the full corruption effect

Correlation, over questions, between a patch's shift and the full-corruption shift:

| Layers | Last position, both | Text KV | Image KV |
|---|---|---|---|
| 0–7 | 0.51 | 0.46 | 0.36 |
| 8–15 | 0.62 | 0.67 | **−0.43** |
| 16–19 | **0.75** | **0.76** | 0.49 |
| 20–23 | −0.30 | −0.17 | −0.34 |
| 24–27 | −0.32 | −0.29 | 0.35 |
| 28–31 | 0.44 | 0.47 | 0.49 |

The patches in 8–19 track the per-question corruption effect (r up to 0.76); the ones in 20–27 are anti-correlated with it.

### Margin structure of selected patches

| Patch | mean $\Delta$ | sd | gap | yes-rate | acc. present | acc. absent |
|---|---|---|---|---|---|---|
| clean | −0.108 | 0.624 | 0.688 | 0.437 | 0.633 | 0.782 |
| corrupted | +0.192 | 0.548 | 0.032 | 0.660 | 0.671 | 0.352 |
| last-pos. heads 8–15 | +0.599 | 0.423 | 0.198 | 0.903 | 0.937 | 0.134 |
| last-pos. both 16–19 | +0.077 | 0.326 | 0.227 | 0.573 | 0.715 | 0.585 |
| text KV 8–15 | +0.182 | 0.479 | 0.328 | 0.637 | 0.759 | 0.500 |
| last-pos. MLPs 28–31 | +2.280 | 1.252 | 1.058 | 0.983 | 1.000 | 0.035 |

Fraction of answers changed relative to the clean model (present / absent): corruption 0.39 / 0.57; text KV 8–15: 0.20 / 0.32; image KV 8–15: 0.04 / 0.04.

---

## 3. How this fits steps 3–6

- **Step 6 / 6x (what is necessary):** mean-ablating MLP windows 16–19 and 16–23 costs the most AUROC ($-0.054$, $-0.071$). Step 7 patching the same last-position MLPs gives a similar AUROC ($0.757$ vs clean $0.818$), so the two tests agree there.
- **Patching vs ablation for heads.** Mean-ablating heads barely changes AUROC (step 6), but patching the heads of 8–15 with the corrupted values drops AUROC to $0.632$ and reproduces 66% of the effect. These are different questions: ablation tests whether the information is *necessary*, patching whether it is *sufficient to carry the corrupted state*. The heads' information is redundant (others can replace it) but is read downstream, so a wrong value misleads.
- **Step 5 (where the effect appears at the last position):** the direct effect builds up in MLPs 16–23 (L18, L20, L23). Step 7 adds that this effect is fed by information routed through text-token positions in layers 8–19, not by direct reading of the image tokens.
- **Late MLPs:** 5d and 6x describe L28–L31 as gain-like (removal widens the margin); step 7 shows the same (bias and spread) and the negative per-question correlation of layers 20–27 fits compensation.

### Overall picture (noise-mode, last-position view)

1. Layers 0–7: no detectable image-dependent signal at the last token or in K/V.
2. Layers 8–19: image information, already moved into the text-token positions, is read by the last token (K/V of text tokens, then last-position heads and MLPs). This is the only stage whose corruption reproduces most of the effect. The yes-bias of the noise run also enters here.
3. Layers 16–23: the evidence MLPs turn this into the yes/no evidence (steps 5 and 6x); mean-ablation of 16–19 hurts most.
4. Layers 20–31: no new image information needed; the late MLPs set the gain and threshold, and partly counter the earlier stages.

---

## 4. Caveats

- **n = 300**, one run, noise corruption only; `shuffle` and `none` are not tested (K/V patching is not available for `none`). Several fractions have wide intervals (0–7, 20–27, 28–31).
- **$\text{frac}$ is a ratio with a small denominator** ($|{-0.311}|$ versus per-question sd $\approx0.7$). It also mixes the loss of evidence with bias: a large mean shift towards Yes also moves the signed shift when the classes are unequal (158 present vs 142 absent). This is small for most windows but makes the 28–31 MLP value ($-1.0$) meaningless as an effect size.
- **Corrupted activations are off-distribution** (noise tokens, yes-rate 0.66), so patching them in transfers a bias as well as missing evidence. A window that "reproduces" the effect is sufficient to carry the corrupted state; it is not shown to be the only place where the evidence lives.
- **"Text-token K/V" includes BOS, the instruction and the question tokens.** The experiment does not say which of these carries the image information.
- Image-token K/V patching only changes what the *last token* reads from the image. It says nothing about how text positions themselves acquire image information in earlier layers.
- Patches in different routes overlap (text KV 8–15 $0.53$ vs last-position both $0.70$; difference $-0.17$ [−0.37, 0.06], not significant), so the contributions are not additive.
- First-order: windows are coarse and non-overlapping; finer windows (e.g. 4 layers in 8–19) were not run. About 36 patch types were tested without correction for multiple comparisons.

---

## 5. Suggested follow-up

1. **Finer windows in 8–19** (4 layers each) for last-position and text-token K/V patching, to locate where the information is moved.
2. **Split the text tokens** into BOS / instruction / question / "ASSISTANT:" to find which positions hold the image information.
3. **Patch the text positions' residual streams, and patch the image K/V in the prefix computation of text tokens**, to see where the text tokens read the image.
4. **Mean-ablate window heads** (8–15, 16–19) to compare directly with the patching result for heads.
5. **`shuffle` run** (K/V patching is available) to check that layers 8–19 and the text-token route hold under an in-distribution corruption.
