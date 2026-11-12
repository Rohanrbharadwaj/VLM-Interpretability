# Step 6: mean-ablation at the last position. Findings and conclusions

**Setting.** JEPA-VLM (I-JEPA ViT-H/16 → projector → Vicuna-7B-v1.5, 4-bit), noise-corruption control ($k=0.5$), 400 balanced POPE questions (odd image ids). Each config replaces the last-position output of the listed components by its mean over 200 reference questions: the question-specific variation is removed and the average offset is kept. $\Delta=\text{logit(Yes)}-\text{logit(No)}$. Baseline: AUROC $0.812$, accuracy $0.700$, yes-rate $0.423$, gap between present and absent $0.661$, $d'=1.28$. All CIs are paired bootstrap 95% intervals; random controls are 10 matched sets per size.

---

## 1. Conclusions

1. **Question-specific evidence at the last token is carried by MLPs, not heads.** Mean-ablating any single head, or any group of heads, changes AUROC by at most $0.011$; second-half heads *raise* AUROC ($+0.024$).
2. **No single MLP is necessary.** The largest single drops are L28.mlp ($-0.015$) and L16.mlp ($-0.012$); L18.mlp, which has 44% of $\mathrm{Var}(E)$ in step 5d, costs only $-0.008$ (CI [−0.015, −0.000]). Direct-effect share did not translate into necessity.
3. **The evidence MLPs matter as a group, beyond chance.** {L16, L17, L18, L20, L23}: $-0.057$ [−0.076, −0.039] vs random 5-MLP sets $-0.006\pm0.009$ (worst random $-0.021$).
4. **Within the MLPs, the evidence sits in layers 16–23, mostly 16–19.** Window 16–19: $-0.054$; 16–23: $-0.071$; both below the minimum of their size-matched random sets. Layers 0–11 and 24–27 contribute nothing at this position. Window 28–31 ($-0.040$) also falls below its random minimum, but with a different signature (§4).
5. **Strong non-additivity.** Ablating the evidence MLPs costs $-0.057$, the other 11 second-half MLPs cost $-0.016$ (random 11-MLP sets: $-0.013$), but all 16 together cost $-0.202$. Interaction $=-0.128$ [−0.165, −0.093]. This is consistent with compensation among MLPs (evidence and non-evidence MLPs absorb each other's removal), which also fits the $-0.76$ per-question correlation between evidence and opposing MLP effects in 5d. It is not proof of self-repair: part of the effect could be generic nonlinear damage from removing many MLP outputs at once.
6. **The second half is specific, beyond generic damage.** Random 16-MLP sets give $-0.078\pm0.042$ (range $-0.028$ to $-0.142$); the second half is $-0.123$ [−0.159, −0.089] below their mean (paired).
7. **Two roles among second-half MLPs.** Evidence MLPs (16–23) produce most of the per-question variation of $\Delta$. The later MLPs (24–31) act mainly as gain control: removing them widens the logit distribution without destroying discrimination (§4).
8. **Opposing MLPs {21, 24, 27} do not carry evidence.** Removing them is neutral or slightly beneficial ($+0.005$; L27.mlp alone $+0.004$ [+0.001, +0.006]).

---

## 2. Singles and groups

| Component or group | $\Delta_{\text{AUROC}}$ [95% CI] | Random sets (mean ± sd; min) |
|---|---|---|
| L28.mlp | −0.015 [−0.024, −0.005] | |
| L16.mlp | −0.012 [−0.021, −0.003] | |
| L17.mlp / L18.mlp | −0.008 / −0.008 | |
| L23.mlp | −0.007 [−0.013, −0.003] | |
| L20.mlp | −0.003 [−0.005, −0.000] | |
| L27.mlp | +0.004 [+0.001, +0.006] | |
| L16H0 | −0.007 [−0.013, +0.000] | |
| all other single heads and MLPs | $\lvert\cdot\rvert\le0.002$, CIs include 0 | |
| **evidence MLPs** (5) | **−0.057** [−0.076, −0.039] | −0.006 ± 0.009; −0.021 |
| opposing MLPs (3) | +0.005 [+0.002, +0.009] | −0.009 ± 0.008; −0.025 |
| answer-reactive MLPs {28, 30} | −0.018 [−0.029, −0.006] | −0.002 ± 0.006; −0.015 |
| evidence heads (4) | −0.011 [−0.022, −0.000] | −0.000 ± 0.001; −0.003 |
| bias heads (5) | +0.002 [−0.001, +0.004] | −0.000 ± 0.001 |
| all heads, first half (512) | −0.030 [−0.050, −0.015] | |
| all heads, second half (512) | **+0.024** [+0.006, +0.042] | |
| all MLPs, first half (16) | −0.036 [−0.050, −0.024] | |
| all MLPs, second half (16) | **−0.202** [−0.243, −0.155] | −0.078 ± 0.042; −0.142 |

With 10 random sets the smallest attainable permutation $p$ is $1/11$, so the strongest statement is that a real group is several times worse than the worst random set, together with the paired-bootstrap interval.

---

## 3. Layer localisation (MLP windows)

| Window | $\Delta_{\text{AUROC}}$ [95% CI] | Random same-size sets (mean ± sd; min) | Below random min? |
|---|---|---|---|
| 0–3 / 4–7 / 8–11 | +0.002 / +0.005 / +0.002 | 4 MLPs: −0.015 ± 0.013; −0.038 | no |
| 12–15 | −0.032 [−0.042, −0.022] | same | no |
| **16–19** | **−0.054** [−0.072, −0.037] | same | **yes** |
| 20–23 | −0.008 [−0.013, −0.002] | same | no |
| 24–27 | +0.004 [+0.000, +0.009] | same | no |
| **28–31** | **−0.040** [−0.056, −0.022] | same | **yes** |
| 0–7 | +0.006 | 8 MLPs: −0.027 ± 0.017; −0.052 | no |
| 8–15 | −0.040 [−0.053, −0.029] | same | no |
| **16–23** | **−0.071** [−0.091, −0.051] | same | **yes** |
| 24–31 | −0.024 [−0.040, −0.008] | same | no |
| second half minus the 5 evidence MLPs (11) | −0.016 [−0.031, −0.001] | 11 non-evidence MLPs: −0.013 ± 0.010; −0.026 | no |

Window 20–23 contains two evidence MLPs (L20, L23) yet costs only $-0.008$. MLPs 12–15 reduce AUROC but stay within the random range.

**Random sets and the evidence MLPs.** Random-set drops scale with set size ($-0.0047$ per MLP) and with the number of evidence MLPs they happen to contain ($-0.0044$ per evidence MLP; $R^2=0.52$; within size 4 and 8 the correlation with the evidence count is $-0.58$ and $-0.67$). That regression predicts $-0.087$ for the real second half (observed $-0.202$) and $-0.041$ for the rest of the second half (observed $-0.016$).

**Additive per-layer model.** A ridge model with one coefficient per MLP layer, fitted on the random sets, special set and step-6 singles/groups, predicts the held-out layer windows with correlation $0.89$. It predicts the evidence group at $-0.073$ (observed $-0.057$), the first half at $-0.039$ (observed $-0.036$), but the second half at $-0.093$ (observed $-0.202$). Large coefficients (rough, 68 configs for 32 parameters): L14 $-0.022$, L15 $-0.031$, L16 $-0.025$, L17 $-0.025$, L28 $-0.018$, L29 $-0.015$; positive for L22 ($+0.021$) and L27 ($+0.011$). L14 and L15 are not in the evidence set and were never ablated alone.

---

## 4. What the ablations do to the logit distribution

| Config | gap | sd | $d'$ | AUROC | accuracy | yes-rate |
|---|---|---|---|---|---|---|
| baseline | 0.661 | 0.516 | 1.28 | 0.812 | 0.700 | 0.422 |
| evidence MLPs (5) | 0.212 | 0.216 | 0.98 | 0.754 | 0.650 | 0.292 |
| MLPs 16–19 | 0.256 | 0.258 | 0.99 | 0.757 | 0.652 | 0.315 |
| MLPs 16–23 | 0.183 | 0.198 | 0.93 | 0.741 | 0.625 | 0.292 |
| MLPs 12–15 | 0.478 | 0.426 | 1.12 | 0.780 | 0.675 | 0.398 |
| MLPs 28–31 | 0.765 | 0.710 | 1.08 | 0.772 | 0.678 | 0.415 |
| second half minus evidence (11) | 1.089 | 0.917 | 1.19 | 0.795 | 0.702 | 0.430 |
| all second-half MLPs (16) | 0.134 | 0.330 | 0.41 | 0.610 | 0.575 | 0.352 |

- **Evidence MLPs / layers 16–23:** removing them shrinks both the gap and the spread by about 60–70% ($d'$ falls only 24–28%). They produce most of the per-question variation of $\Delta$, only part of which is signal. Errors shift toward misses (accuracy on present objects $0.445$ for the evidence group, on absent objects $0.874$).
- **Layers 28–31 and the other second-half MLPs:** removing them *widens* the distribution (sd $0.516\to0.71$ and $0.92$) while $d'$ drops modestly. They scale the logit difference down; their removal is a gain increase, not a loss of evidence.
- **Removing both** leaves $d'=0.41$ and AUROC $0.61$: what remains at the last position is mostly noise plus whatever signal reaches it through earlier layers.

---

## 5. How this fits the earlier steps

- **Step 5d:** the evidence-MLP group discriminates better than $E$ (AUROC $0.780$ vs $0.728$, difference $+0.052$ [+0.014, +0.089]); L18 explains 44% of $\mathrm{Var}(E)$ (CI [0.41, 0.47]). Step 6 confirms the group but not L18 alone.
- **Compensation:** step 4 showed large compensation for zero-ablation (L20 about 87%, L28 about 45%); 5d showed evidence–opposing coupling of $-0.76$ (absent) and $-0.68$ (present); step 6x shows an interaction of $-0.128$. These are consistent with a compensating structure; none isolates it as the cause.
- **Step 4 correction:** the AUROC drops of the grounding top-5/top-10 zero-ablations (0.765, 0.722) are matched by random sets (0.755–0.763 and 0.607–0.798). Step 4 therefore shows no pathway-specific discrimination effect; only the bias shifts are pathway-specific. See the correction block in `step4d_results.md`.
- **L28.mlp:** zeroing it makes the model answer "yes" to everything (step 4), and mean-ablating it costs $0.015$ AUROC (step 6). It sets the threshold and also carries a little per-question signal.

---

## 6. Caveats

- Last-position ablation only. After removing all second-half MLPs at the last position AUROC is still $0.61$, so evidence reaches the last token by other routes: earlier layers' MLPs (12–15 are partly implicated) and attention over image tokens at earlier positions. Step 7 (K/V patching) addresses this.
- $n=400$ questions, 300 bootstrap resamples (for 6c/6x), noise-mode only. `none` and `shuffle` are untested.
- Random sets are a conservative control for the second-half claim, because they contain 1–4 evidence MLPs by chance.
- About 30 single and group tests were run; borderline results (L18, L16H0, evidence heads) should not be cited as effects.
- The additive layer model is fitted on few configs; use its coefficients only to choose what to test next.
- "Compensation" and "gain control" are interpretations of the logit-distribution changes, not causally isolated mechanisms.

---

## 7. Suggested next checks

1. **Interaction curve:** ablate the evidence MLPs plus $k\in\{2,4,6,11\}$ of the other second-half MLPs (several random draws) to see how fast discrimination collapses and which MLPs act as the repairers.
2. **New singles:** L14, L15, L22, L29, L31 (flagged by the additive model but never ablated alone).
3. **Step 7:** K/V and last-position patching by layer window, for the route that last-position ablation cannot see.
4. **Other control modes** (`none`, `shuffle`): check that layers 16–19 remain the main evidence site.
