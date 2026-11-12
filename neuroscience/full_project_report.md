# Evidence versus prior in LLaVA-1.5-7B, compared with mice, and its link to POPE hallucination

**Full project report: everything from the main run through notebooks A, B, B2, B4, C, D, D2, E, F, F2a and F2b, with questions Q1–Q6.**
Author: Ponish Bhatia (IIT Kanpur, UGP / Track B). Model: `llava-hf/llava-1.5-7b-hf`, 4-bit (bitsandbytes), 2×T4 on Kaggle. Mouse data: International Brain Laboratory (IBL) behaviour via ONE / openalyx.

---

## 0. One-page summary

**The question.** In the IBL task a mouse sees a Gabor grating on the left or right and must report the side. The prior is the block structure: for blocks of ~20–100 trials the stimulus is on the right 80% (or 20%) of the time. We built an analogue for LLaVA: the grating is the evidence, and a sentence in the prompt ("In the last 10 trials the grating was on the left X times and on the right Y times.") is the prior. We ask how the two are combined (Q1, Q2), where they live and how they reach the answer (Q3), what happens on conflict trials (Q4), how this compares with mice (Q5), and whether the in-context prior shares machinery with the *learned* prior behind POPE hallucination (Q6).

**Answers, in one line each (details and caveats in §12).**

| Q | Answer |
|---|---|
| Q1 | Yes. Δ = b + βs + γπ fits with γ = 0.277 [0.233, 0.322] (history prompt); no-history control γ ≈ 0. Screen 0.227 → full run 0.277 (counts10 was the best of five formats; none met the pre-set pass rule, §3). |
| Q2 | Roughly an additive offset: the side-matched prior effect is ≈ +0.75 logits, flat across contrast. Mouse: no consistent contrast dependence (low power); mouse lapses also move with the prior (D2). |
| Q3 | The count is linearly readable from layer ~8 (corr 0.95 from L16). It reaches the answer through digits → sentence tokens → answer position (L16–20). Steering effects are mostly explained by overlap with the *evidence* axis; at most a small, borderline count-specific remainder. The history sentence halves the evidence swing (ratio ≈ 0.4–0.55). |
| Q4 | One rule fits conflict and congruent trials: Δ_hist ≈ 0.43·Δ_none + 0.63·sign(count) − 0.11 (gain loss + offset). Mice are 0.93 (congruent) vs 0.87 (conflict) at 100% contrast, and 0.78 vs 0.53 at 6%. |
| Q5 | LLaVA's prior boundary shift (12.2% contrast, history β) lies inside the mouse spread (median 9.6% [5.4, 15.0]); both look additive; mice follow the recent-10 count better than the true block (block model wins in 10% of mice). "Not distinguishable", not "matching". |
| Q6 | No: the in-context count and the learned (POPE) prior behave as **separate levers**. Count effect on POPE is +0.09 logits vs +0.66 on gratings, no split interaction, not a wording artefact; mechanistically the count is routed via question tokens, is not aligned with the learned-prior axis, and does not act as a "yes" lever. Moderate evidence (one model, small mechanistic samples). |

---

## 1. Background and motivation

- **IBL mouse task.** A Gabor patch of contrast c ∈ {0, 3%, 6.25%, 12.5%, 25%, 50%, 100%} appears left or right; the mouse turns a wheel. After a 90-trial unbiased start, blocks alternate with P(right) = 0.2 or 0.8. Mice use the block prior mainly when the stimulus is weak.
- **Predictive-coding link.** The IBL "brain-wide map of a prior" paper (Findling et al.) shows the prior is decodable before the stimulus in many regions, from sensory to motor. The analogue question for a VLM: is the prior combined with evidence at the answer, or does it change how evidence is encoded?
- **Link to hallucination.** POPE-style hallucination (saying "yes" to an absent object) is a learned prior (object co-occurrence) beating weak visual evidence. If an *in-context* prior used the same machinery, in-context experiments would be a controlled model of hallucination (Q6).
- **UGP context.** The UGP proposal (`task_document.pdf`) is about attribute/relation hallucination and a JEPA encoder swap; this mouse thread is a side contribution that touches it through POPE. Project reference numbers (`encoder_swap_final_results.md`): 1000 balanced POPE questions, LLaVA truthful margin 5.23 vs JEPA 0.67, evidence MLPs at L16–23, route through text-position K/V at L8–19. A related paper in the project is arXiv 2605.13156 (dual-pathway circuits of object hallucination).

## 2. Common definitions

- **Δ** = logit(Right) − logit(Left) at the first answer token (gratings; averaged over two wordings in the main run). For POPE: Δ = logit("▁Yes", id 3869) − logit("▁No", id 1939).
- **Behavioural model:** Δ = b + β·s + γ·π, fitted at contrast ≤ 0.09, where s = signed contrast (right positive) and π = log-odds of the prior. Boundary shift (contrast units) = 2.77·γ/β.
- **Swing** = mean Δ(right grating) − mean Δ(left grating) at contrast ≥ 0.09 (or ≥ 0.125). **Swing ratio** = history swing / no-history swing (1 = no evidence lost).
- **Gain + offset rule:** Δ_hist ≈ a·Δ_none + c·sign(count) + b.
- **Hidden-state convention:** hidden_states[L] = output of decoder layer L−1; hooks sit on LAYERS[L−1]; they move the vector to the layer's device (4-bit model split over two GPUs).
- **Prompt (counts10):** after the image, before the question: "In the last 10 trials the grating was on the left X times and on the right Y times." It carries only the count, not the order or the block.
- **Statistics:** image-clustered or trial-block bootstraps; sign-flip and permutation nulls; deterministic forward passes (~0.66–0.7 s each).

## 3. Part 0 — Stimulus pilot, history pilot, format screen and the main run (`prior_vs_evidence_llava.ipynb`)

### 3.0 Pilot stages (before the main run)

**Stage P0 — can LLaVA read a grating at all? (stimulus and prompt screen; 648 forwards, 15.4 min).** Three stimuli (small Gabor, big Gabor, a plain disk as an easy control) × three prompts (`plain+oneword`, `plain+llava`, `vicuna+llava`), contrasts 0.06, 0.125, 0.25, 1.0, 8 trials per side per cell. Probability mass on Left+Right was 1.0 in every row, blank-image Δ −0.5 to −0.8 (a left bias), and every combination was "usable" (balanced accuracy 1.0 at contrasts 0.25 and 1.0).

| Stimulus | Prompt | Swing @0.25 | Swing @1.0 |
|---|---|---|---|
| gabor_small | plain+oneword / plain+llava / vicuna+llava | 3.62 / 3.75 / 2.76 | 3.38 / 3.43 / 2.64 |
| gabor_big | same three | 4.61 / 4.51 / 3.51 | 3.42 / 3.44 / 2.70 |
| disk | same three | 7.32 / 7.15 / 7.71 | 8.07 / 7.95 / 9.17 |

Gabor small, `plain+oneword`: balanced accuracy 0.88 at 6% contrast (P(Right | right stimulus) 0.75), 1.00 from 12.5%; mean Δ for left vs right stimuli at 6%: −2.01 vs +0.18. A finer contrast sweep (1%–6%, blank Δ −0.72) shows the psychometric threshold: swing −0.05 (1%), 0.19 (2%), 0.49 (3%), 0.99 (4%), 1.56 (5%), 2.08 (6%), balanced accuracy 0.50 until 3%, 0.68 at 5%, 0.82 at 6%. So the model genuinely detects a weak grating around 4–6% contrast, as mice do. The prompt wording changed the swing by up to ~30% (e.g. gabor_small @0.25: 3.62 vs 2.76); all were usable.

**Stage P1 — first history pilot (hard "Left/Right" answers; history placed before vs after the image, with a `none` control).** A generated one-word answer was scored, and P(answer right) plotted against signed contrast for block P(right) 0.2 vs 0.8. Findings:
- Placing history **after** the image: accuracy at 100% contrast 0.609, γ ≈ +0.08–0.10, P(answer right) at 0% contrast 0.30 (block 0.2) vs 0.65 (block 0.8); the red (0.8) curve sat above the blue (0.2) curve but the curves were noisy.
- Placing history **before** the image: accuracy 0.669, P(answer right) at 0% contrast 0.0 in both blocks (the model almost always said Left), so no prior could be seen in hard choices.
- Hard choices were too coarse (saturated at 0/1), which is why all later work uses the continuous logit difference Δ.
- The `none` control gave a block effect of 0.02, the null we want.

**Stage P2 — history pilot with continuous Δ (`pilot2.csv`, 1800 rows = 600 trials × {none, before, after}; 37.8 min, ≈15.9 trials/min, mass 1.00).** Gabor with contrast 0–1, block P(right) 0.2/0.5/0.8, history up to 30 trials with the count of "right" recorded.
- Swing at contrast ≥ 0.09: none 3.23, **after 1.61 (ratio 0.50), before 0.67 (ratio 0.21)**.
- Mean Δ at contrast ≤ 0.09 for block 0.2 / 0.5 / 0.8: after −1.10 / −0.77 / −0.45; before −1.75 / −1.56 / −1.40.
- Side-matched effect of block 0.8 vs 0.2 on Δ: **before +0.22 [0.15, 0.29] (γ-equivalent +0.081), after +0.30 [0.21, 0.36] (γ-equivalent +0.107)**; per-contrast values 0.17–0.36 with wide CIs at 4% and 9% (few trials per cell). Fits: γ = +0.071 (before) and +0.099 (after), CIs excluding 0.
- Blank image (pure prior): block 0.8 minus 0.2 = +0.25 (before), +0.34 (after); `none` 0.02.
- **A mistake in my first analysis cell (Section [4]):** it printed negative prior effects (−0.80, −0.43). Cause: block 0.8 has more right-side trials and history compresses right-side Δ more than left-side Δ, so pooling mixed side composition into the "block" effect. Splitting by side and contrast gives a paired effect of about +0.2 to +0.4 in both orders, similar at 100% and at low contrast; with only 3–9 trials in many cells I claimed only "no sign of contrast dependence yet".
- **Reading of the pilot:** the prior exists but is weak (γ ≈ 0.07–0.10 over a 2.77 log-odds gap is only 0.2–0.3 logits; an ideal base-rate user would have γ ≈ 1, so LLaVA uses it at ~10%) and it costs half or more of the evidence. Phase-1's gate (reads the grating; γ nonzero with CI excluding 0) passed on paper, but I advised against caching residuals yet, since the model loses half the evidence for ~10% of the benefit. This motivated the format screen below.

**Pre-registered rule for the screen (fixed before seeing the results):** a format passes if swing ≥ 70% of no-history swing and γ ≥ 0.25 with a 95% CI above 0. I also said that if every format failed, the honest conclusion would be that LLaVA-1.5-7B cannot carry both knobs at once, and the next move would be a different model or the object-presence (POPE) variant. That is part of why POPE (F) was later added.

### 3.1 Process (format screen and main run)
1. **Format screen** (300 trials each, `none` plus five formats). Variants tried: counts over 30 trials, probabilities over 30 trials (e.g. "about 73% of the time"), each with and without a short instruction ("If the image is unclear, use this information"), and counts over a shorter 10-trial window.

| Format | Swing ratio | β | γ [95% CI] | Passes pre-set rule |
|---|---|---|---|---|
| counts10 | 0.533 | 6.35 | 0.227 [0.180, 0.273] | **No** (swing < 70%, γ < 0.25), but best of the five |
| counts30 | 0.492 | 6.16 | 0.118 [0.085, 0.145] | no |
| counts30 + instruction | 0.317 | 4.05 | 0.170 [0.136, 0.200] | no |
| prob30 (probabilities) | 0.317 | 4.56 | −0.010 [−0.050, 0.020] | no |
| prob30 + instruction | 0.253 | 4.35 | −0.018 [−0.054, 0.011] | no |

   **No format passed the pre-set criteria.** Verbal probabilities carried no prior at all; counts did, and the short window was strongest. I selected counts10 as the least bad format, i.e. the main run proceeds *despite* the gate failing; the swing loss and weak-ish γ are therefore a known feature of this model, not a tuned success. (An earlier version of this report wrongly labelled counts10 as "passes".)
2. **Main run:** 600 trials × [none, hist] prompts × 2 wordings (1200 rows), all contrasts, residual states cached at the last position for 33 layers (`full_hs_*.npy`, fp16, 600 × 2 × 33 × 4096). Log: natural behavioural half-jump (high-count minus low-count)/2 = 0.33 logits; natural half-jump norm vs typical residual norm per layer: L8 (0.05, 9.3), L10 (0.11, 12.8), L12 (0.61, 17.1), L14 (1.09, 20.7), L16 (1.87, 27.3), L20 (3.18, 54.2), L24 (3.66, 87.6); cos(count, evidence) at L8/12/16/20 = −0.01 / 0.06 / 0.42 / 0.69.

### 3.2 Results (`summary.txt`)
- Median probability mass on the two answer tokens 0.999.
- **Swing by contrast (none → hist, ratio):** 3% 0.56→0.76 (1.35); 4.5% 1.21→0.81 (0.66); 6.25% 2.11→1.15 (0.54); 9% 2.74→1.58 (0.58); 12.5% 3.40→1.85 (0.54); 100% 3.44→1.97 (0.57). Side bias (blank image): none −0.73, hist −0.55.
- **Fit Δ = b + βs + γπ (contrast ≤ 0.09):**
  - no history: b = −0.85, β = 15.0 [14.4, 15.6], γ = −0.003 [−0.023, +0.021] (0.2-vs-0.8 gap −0.01 logits);
  - history: b = −0.59, β = 6.3 [5.1, 7.5], **γ = +0.277 [+0.233, +0.322]** (0.2-vs-0.8 gap +0.77 logits).
- **Side-matched prior effect** (history minus none, block 0.8 minus 0.2): +0.75 logits [+0.64, +0.86] (γ-equivalent 0.270). Per contrast: 0: 0.76; 3%: 0.80; 4.5%: 0.85; 6.25%: 0.49 (CI crosses 0); 9%: 0.84; 12.5%: 0.87; 100%: 0.64 — flat.
- **Boundary shift:** γ/β = 0.0439 → 12.2% contrast for the 0.2-vs-0.8 gap; with the no-history β it is ≈5%; shift / half-swing contrast (5.5%) = 2.22; prior push / full evidence swing = 0.223.
- **Probes (answer position, grouped CV by block; PCA-32 per layer):** side AUROC ≈ 1.0 from layer 2 in both runs; block AUROC in the history run 0.56 (L2), 0.66 (L4), 0.82 (L6), 0.84 (L8), 0.86–0.88 from L10; no-history control 0.48–0.57; permutation null ≈ 0.5.

## 4. Part A — cache analysis (`A_cache_analysis.ipynb`, 75 s CPU, plus two patch cells run afterwards as separate CPU cells)

**Question:** is the prior linearly readable, does the readout explain per-trial behaviour beyond the count, and how does it sit relative to the evidence direction?

- **A1 — count direction (out of fold; correlation with count after removing side × contrast cells):** 0.05 (L0), 0.32 (L8), 0.63 (L10), 0.75 (L12), 0.85 (L14), 0.95 (L16), 0.96 (L18), ≈0.94 thereafter. No-history control max |0.13|. Block AUROC 0.90 = the ceiling of a count-only readout.
- **A2 — does the projection explain the behavioural shift beyond the count?** The raw run showed partial correlations of up to +0.34 at L28 (dR² ≤ +0.017). **This was an artefact:** the behaviour is a step in the count (flat for 1–7, jump at 8–10), and a linear count baseline leaves a step-shaped residual. With a one-hot count baseline (patch cell, layers 10–15 checked) the effect is ≈ 0. Retracted.
- **A3 — linearity:** representation R² linear vs free-per-count: 0.70 vs 0.77 (L16–24); behaviour 0.13 vs 0.25. Mean shift by number of "right" in the prompt: 0: −0.84; 1: +0.25; 2: +0.02; 3: −0.10; 4: −0.03; 5: −0.01; 6: +0.41; 7: +0.08; 8: +0.43; 9: +0.58; 10: +1.31. The step is downstream of the (linear) representation.
- **A4 — geometry (unmatched; superseded for the onset layer by §4c):** cos(count direction, evidence direction) rises 0.0 (≤L12) → 0.42 (L16) → 0.55 (L18) → 0.69 (L20–26) → 0.76 (L28). d′ of right vs left along the no-history evidence axis: none ≈ 10.5–11.8 vs hist ≈ 3.6–4.6 from L16, ratio ≈ 0.34–0.42 (0.29 at L12). A matched-count patch cell showed part of the apparent early (L11–13) ratio was a side/count confound; the order of onset is therefore unclear.
- **Conflict (Q4a):** VLM conflict vs congruent margin gap ≈ 0.5–0.85 logits at every contrast: a constant offset.

### 4b. A-patch 1 — one-hot-count baseline for A2, and a look at layers 10–15 (CPU, ~1–2 min)
Rationale: behaviour is a step in the count, so a smooth-count baseline leaves a step-shaped residual that a count-correlated direction can "explain". The patch removes cell effects (side × contrast) **plus one free value per count 0–10** before correlating the out-of-fold projection z with the per-trial shift (287 full-window trials; 5-fold grouped CV by block; 150 block-bootstrap draws).

Partial correlation of z with the shift (old = smooth count; new = one-hot count; new CI):

| Layer | old | new | new CI | dR² (new) |
|---|---|---|---|---|
| 6 | −0.142 | 0.005 | [−0.074, 0.075] | −0.001 |
| 8 | −0.192 | 0.005 | [−0.127, 0.089] | −0.001 |
| 10 | −0.271 | −0.021 | [−0.186, 0.105] | −0.003 |
| 12 | −0.465 | −0.016 | [−0.172, 0.091] | −0.003 |
| 14 | −0.345 | −0.078 | [−0.223, 0.037] | −0.003 |
| 15 | −0.276 | −0.113 | [−0.249, −0.005] | −0.001 |
| 16 | 0.126 | −0.114 | [−0.264, −0.003] | −0.002 |
| 18 | 0.256 | −0.081 | [−0.235, 0.038] | −0.002 |
| 20 | 0.256 | −0.059 | [−0.225, 0.058] | −0.002 |
| 24 | 0.267 | −0.070 | [−0.227, 0.049] | −0.002 |
| 28 | 0.343 | −0.073 | [−0.226, 0.040] | −0.002 |
| 32 | 0.167 | −0.077 | [−0.194, 0.013] | −0.001 |

(Layers 7, 9, 11, 13, 17, 19, 21–23, 25–27, 29–31 are similar: new ≈ −0.12…+0.02.) Only layers 15–16 have an interval excluding 0, marginally negative; with 27 layers tested I do not interpret them. **Conclusion: the earlier "the count direction explains the per-trial shift beyond the count" (+0.26 to +0.34 at L18–32) was a step-shape artefact and is retracted.** The negative values at L8–15 were the same artefact. This is a null with little power (little shift variance is left once cells and count are removed); a correlation cannot show use anyway — that needs interventions (notebook B).

**Layers 10–15, is there an outlier dimension?**

| Layer | mean norm none | mean norm hist | top-3 dims | top-3 variance share | evid. ratio (all dims) | without top-3 |
|---|---|---|---|---|---|---|
| 10 | 12.78 | 12.14 | 2235, 2789, 3431 | 0.03 | 0.89 | 0.89 |
| 11 | 14.75 | 14.16 | 1512, 2235, 2789 | 0.03 | 1.13 | 1.13 |
| 12 | 17.06 | 16.39 | 1512, 2235, 2789 | 0.01 | 1.79 | 1.79 |
| 13 | 18.67 | 18.08 | 1512, 2235, 2789 | 0.01 | 0.67 | 0.67 |
| 14 | 20.68 | 19.88 | 1512, 2789, 3431 | 0.01 | 0.53 | 0.53 |
| 15 | 25.61 | 24.90 | 2789, 1512, 3431 | 0.01 | 0.43 | 0.43 |

Norms rise smoothly and the top-3 dimensions hold ≈1% of the variance, so layer 12's 1.79 is **not** an outlier-dimension problem; it is a real value of that (unmatched) statistic, caused by the confound below.

### 4c. A-patch 2 — evidence separation measured within each count value (CPU, ~1–2 min)
Because side and count are correlated through the blocks, the right-minus-left difference in the history run also contains the count signal. This cell computes right-minus-left separation **within each count value** (weights nR·nL/(nR+nL); 287 trials with contrast ≥ 0.09 and a full 10-trial window, 134 right / 153 left; every count 0–10 had ≥ 3 trials on each side) and reports the history-run signal relative to the no-history run along the no-history evidence axis (axis_ratio; 1 = no suppression), 100 bootstrap draws.

| Layer | norm ratio | axis ratio | CI |
|---|---|---|---|
| 1 | 0.75 | 0.72 | [0.66, 0.79] |
| 2 | 0.83 | 0.76 | [0.72, 0.79] |
| 4 | 0.90 | 0.79 | [0.72, 0.84] |
| 5 | 0.82 | 0.69 | [0.64, 0.77] |
| 6 | 0.79 | 0.62 | [0.57, 0.69] |
| 8 | 0.81 | 0.54 | [0.48, 0.64] |
| 9 | 0.82 | 0.50 | [0.45, 0.59] |
| 10 | 0.74 | 0.37 | [0.33, 0.44] |
| 11 | 0.77 | 0.38 | [0.36, 0.43] |
| 12 | 0.83 | 0.33 | [0.26, 0.44] |
| 13 | 0.41 | 0.29 | [0.27, 0.30] |
| 14 | 0.50 | 0.44 | [0.43, 0.46] |
| 15 | 0.38 | 0.30 | [0.28, 0.31] |
| 16 | 0.40 | 0.30 | [0.29, 0.31] |
| 18 | 0.45 | 0.36 | [0.34, 0.38] |
| 20 | 0.46 | 0.39 | [0.38, 0.41] |
| 24 | 0.46 | 0.38 | [0.37, 0.40] |
| 28 | 0.44 | 0.38 | [0.37, 0.40] |
| 32 | 0.51 | 0.43 | [0.41, 0.45] |

(Layers 3, 7, 17, 19, 21–23, 25–27, 29–31: 0.79, 0.58, 0.37, 0.40, 0.40, 0.40, 0.39, 0.38, 0.38, 0.38, 0.38, 0.38, 0.38.)

Findings:
1. The earlier unmatched numbers were confounded; measured within count the history-run evidence is **≈0.3–0.4 of the no-history run from layer 10 on** (earlier estimate 0.53 was too high), and the layer-12 outlier (1.79) disappears.
2. **The suppression is gradual and starts at layer 1, not at L12–14:** 0.72–0.79 at L1–4, falling from 0.69 (L5) to 0.37 (L10), flat ≈0.3–0.4 afterwards. At L6–9 the count direction barely exists yet (corr 0.08 at L6, 0.32 at L8) while evidence is already down to 0.5–0.6, so **the count arriving cannot be what suppresses the evidence; "suppression begins where the count arrives" is retracted.**
3. **It does not depend on the count.** Behavioural swing ratio (history swing / no-history swing, contrast ≥ 0.09) by count in the prompt: ≤2 right 0.50; 3–4: 0.48; 5 (balanced): 0.42; 6–7: 0.46; ≥8: 0.40. A balanced 5-of-10 sentence carries no prior yet still cuts the swing by more than half, so the loss is a **cost of having that sentence in the prompt**, not a conflict between prior and evidence (slightly worse at extreme counts, CIs overlap).
4. The residual-stream ratio (0.3–0.4) is lower than the behavioural swing ratio (0.4–0.5); the logit read-out is probably nonlinear or saturating, so they should not be equated one-to-one.

**Consequences:** (i) "prior–evidence interference" is not a clean finding; it is a format cost (extra tokens, distance, or diluted attention to the image), with the +0.75 logit prior effect sitting on top; (ii) for the mouse comparison, γ/β inside the history condition (12.2% shift, β = 6.3) is the matched number and the no-history β (5%) mixes two prompts — report 12% as primary and 5% as a bound, with the format cost stated; (iii) test C in B (does the prior direction cut the swing?) should come out null, and it did (§7). These two patches also set the patching range for B (layers 8–28) and led to the B0 wording screen (§5.2).

## 5. Part B — patching and steering (`B_patching_steering.ipynb`, ≈1.5–2 h)

### 5.1 Design and fixes
Self-tests (all passed): reproduction of cached Δ (max |diff| 0.000), zero-vector steering 0.000, own-state patching 0.000, hook vs cached state (cos 1.0000), adding the evidence direction (half residual norm) at L20 moves Δ by +4.30. Fixes: cuda:0/cuda:1 hook mismatch; duplicate base rows on resume; QUICK run (natural-size steering at L ≤ 12 was vacuous) → full run with sizes 1× and 4× at layers 12–24.

### 5.0 The QUICK run of B (smoke test, 20 images; `B_out_quick.zip`) and why B was redesigned
Run first with QUICK=True. Self-tests 1–4 passed; **self-test 5 was invalid** (adding 3× the evidence direction at L12 moved Δ by −0.02, because the natural half-jump norm at L12 is tiny, 0.61 vs residual norm 17); the redesigned test uses L20 and half the residual norm (+4.30).
- **B0 (20 images; swing 3.15 without history):** count 5-of-10 ratio 0.44; short count 0.71; neutral same-length sentence **1.13** (extra text does not hurt); count before image 0.20.
- **B1 patching (24 extreme-count trials, mean shift +0.21):** recovery −0.00 (L8), −0.09 (L12), 0.39 (L16, CI −1.29…1.47) — too few trials to say more. Swing ratio after patching 0.99 / 0.92 / 0.64 at L8 / L12 / L16 (history alone 0.51, n = 14).
- **B2 steering at natural norm:** shift −0.00 at L10 and −0.10 at L14; evidence direction +0.00 and +0.23; random ≈ 0 — natural-size steering at L ≤ 12 is vacuous.
These findings led to: 80 images, patching at nine layers (L8–28), steering at 1× and 4× on L12–24, shuffled-label and random controls, and the expanded wording screen (§5.2).

### 5.2 B0 — wording screen (80 images; swing ratio, prior gap k8 − k2)

| Variant | Swing ratio |
|---|---|
| none | 1.00 |
| neutral sentence | 1.17 |
| sentence before image (k=5) | 0.20 |
| sentence 2 / 5 / 8 | 0.48 / 0.40 / 0.37 |
| short 2 / 5 / 8 | 0.74 / 0.75 / 0.65 |
| letters 2 / 5 / 8 | 0.71 / 0.70 / 0.73 |
| sequence 2 / 5 / 8 | 0.82 / 0.81 / 0.82 |

Prior gap per wording: sentence **+0.86** [0.82, 0.90]; short +0.33; letters +0.29; sequence +0.02. **No wording is free of a swing loss, and the wordings with the largest prior lose the most evidence.** The neutral sentence does not hurt (ratio 1.17), so the cost is the *content* of the count, not extra text (an earlier "any extra text costs evidence" claim was overturned).

### 5.3 B1 — whole-residual patching (160 trials)
Fraction of the history effect on Δ carried by the last-position state: ≈ 0 at L ≤ 12; 0.22 (L14); 0.53 (L16); 0.84 (L18); ≈ 1.0 from L20 (reverse: 0.26 / 0.54 / 0.70 / 0.92). Swing ratio after patching: 0.99 (L8), 0.98 (L10), 0.90 (L12), 0.73 (L14), 0.65 (L16), 0.72 (L18), ≈0.62 from L20. Gain loss (L12–16) therefore arrives before the prior offset (L16–20). Pure prior-shift recovery on extreme counts (110 trials, mean shift +0.16 logits) is noisy; most CIs include 0.

### 5.3b B1 conflict check (Q4c; 19 conflict, 91 congruent; suggestive)
Same patching curve for both groups (~0.5 at L16, ~1.0 from L20); sign-aligned history shift +1.32 conflict vs −0.08 congruent. On all 325 extreme-count trials one rule fits both: Δ_hist = 0.43·Δ_none + 0.63·sign(count) − 0.11 (conflict residual +0.04, congruent −0.01).

### 5.4 B2 — steering (effect = [Δ(+d) − Δ(−d)]/2; natural behavioural half-jump 0.33 logits)
At 1× natural size, layers 12/14/16/20/24: shift +0.01/−0.09/+0.11/+0.25/+0.23; prior +0.00/+0.02/+0.31/+0.48/+0.48; evidence +0.01/+0.21/+0.35/+0.54/+0.52; shuffled-label −0.01/−0.06/−0.30/−0.48/−0.45; random +0.00/−0.01/0.00/0.00/0.00. At 4×: L20 shift +1.01, prior +1.95, evid +2.20, shuf −1.94, random +0.01. Layers ≤ 12 do nothing; the evidence-like directions are strong from L14. Adding the prior direction does **not** cut evidence (swing ratio 0.97–1.02 at 1×; 0.8–1.1 at 4×): C (below).

## 6. Part B2 — controls and token-position patching (`B2_controls_and_token_patching.ipynb`)

Fix: `readout_axis` needed `.detach()`. Self-tests passed (digit tokens [599, 606] decode to '2','8'; delta(2 of 10) −0.55, delta(8 of 10) +0.31; patching all post-image positions at L4 reproduces +0.31; own-state patch 0.000).

- **Which axis does steering use?** Cosines with the unembedding read-out axis: shift 0.01 / 0.11 / 0.10 (L16/20/24); with the evidence axis: −0.17 / 0.21 / 0.14; prior direction with evidence 0.42 / 0.69 / 0.68. Six shuffled-label "shift" directions: their steering effect is predicted by cosine with the **evidence axis** (corr 0.88 / 0.98 / 0.97 across layers; slope = the evidence direction's own effect), not the read-out axis (|cos| ≤ 0.19). *My earlier statement that steering acts through the read-out axis was wrong and was corrected.*
- **After removing the evidence part:** the real shift keeps +0.13 to +0.17 logits (~45% of the 0.33 natural), vs null residual sd 0.07–0.09 (≈ 2 sd with 6 nulls). Evidence-orthogonal shift and count directions give +0.14 to +0.18. Verdict: at most a small, borderline count-specific component (optional B2c with ~20 evidence-orthogonal nulls at L20 not run).
- **B3 token-position patching (40 images, 2-of-10 vs 8-of-10, same token length; natural gap +0.86):** forward recovery — digits 0.60 (L4), 0.18 (L8), 0.16 (L10), 0.09 (L12), 0.05 (L14), ≈0 from L16; sentence words 0.12 (L4), 0.74 (L8), 0.86 (L10), 0.65 (L12), 0.48 (L14), 0.07 (L16); last position −0.00 (L4–12), 0.30 (L16), 0.73 (L18), 0.88 (L20), 0.91 (L24); all post-image positions 1.0. Reverse patching agrees. Chain: **digits → sentence tokens (L4–8) → answer position (L14–20).**

## 7. Part C — does the prior cut the evidence? (inside B)
Adding the prior direction leaves the evidence swing intact (ratio 0.97–1.02 at 1×), whereas patching the history *state* cuts it to 0.62. The suppression travels with the history state, not along the count direction. Tracker q3i.

## 8. Part B4 — conflict gain/offset (`B4_conflict_gain_offset.ipynb`)
Built as an optional extension with more conflict trials (embeds the 160 already patched trial ids). **Not run**; the B-patching rows (19 conflict, 91 congruent) and the 325-trial main-run rule already answer Q4c at suggestive strength.

## 9. Part D — mouse behaviour (`D_ibl_mouse_behavior.ipynb`) and D2 lapses (`D2_mouse_lapses.ipynb`)

### 9.1 D
31 mice, ~35k trials (median 932 per mouse). Login fix: Alyx account plus Kaggle secrets IBL_USER / IBL_PASS.
- β median 16.1 [13.1, 19.2] (IQR 11.5–26.5); **γ median 0.611 [0.534, 0.760]**; γ/β 0.035 [0.019, 0.054]; **boundary shift (0.2 vs 0.8) 9.6% [5.4, 15.0]**; shift / c75 1.54 [1.35, 1.92]; lapses L 2.5% [1.4, 5.9], R 5.4% [2.5, 8.6].
- Prior term preferred (ΔAIC > 0) in 90% of mice.
- M3 (contrast-dependent γ): γ1 = +0.44 [−1.54, +1.27], preferred in 29% of mice → no consistent contrast dependence (low power).
- M5 (both regressors): γ_block +0.179, γ_recent10 +0.675; block model beats recent-10 in 10% of mice → mice follow recent history.
- At 0% contrast mice choose the block-favoured side on 66.5% of 6246 trials.
- **Accuracy congruent vs conflict** (table columns were mislabelled in the notebook output; corrected): 6.25% 0.78 vs 0.53; 12.5% 0.86 vs 0.70; 25% 0.92 vs 0.80; 100% 0.93 vs 0.87. (The 50% row is a different task variant and is ignored.)

### 9.2 D2 — block-dependent lapses
λ_lo = error rate on left stimuli (choose right), λ_hi = error rate on right stimuli; modelled 0.49·σ(a + d·u). Expected signs: d_lo > 0, d_hi < 0 in a right-biased block.
- Model-free at 100% contrast: errors 6.7% vs 13.5% (left stimulus, left vs right block); 8.1% vs 13.0% (right stimulus, right vs left block); per-mouse accuracy gap congruent − conflict +4.7 points [1.9, 9.6], 24/31 mice.
- Pooled LR 205 (block) / 363 (recent-10) on 62 df, p < 1e-16. Errors on left stimuli +2.5 [0.0, 6.0] / +4.7 [0.8, 7.0] points in a right block; on right stimuli −1.7 / −1.7 (CIs touch 0). Expected sign in 22/31 (block) and 28/31 (recent-10) mice. Per-mouse evidence is weak (median ΔAIC −0.8 / +1.8): a population result. Boundary shift unchanged (9.6 → 9.1%; recent-10 11.3 → 9.4%). Robust to clean sessions only.
- LLaVA analogue at 100% contrast: margin toward truth +1.72 (0/118 errors) without history; with history +1.13 congruent (0/81), +0.54 conflict (1/25): the whole distribution shifts (compression-like), not a lapse.

## 10. Part E — neural decoding
**Not run; answered from the literature** (Findling et al., Nature 2025; numbers from the preprint, check the Nature version before quoting): prior decodable before the stimulus in ~21–31% of regions (47/159 ephys, 29.5%), from early sensory (VISp, LGd) to motor/midbrain, surviving video and eye-position controls; mice track recent actions (kernel decay ≈ 5.6 trials) better than the Bayes-optimal block model, matching our D result. Difference: in mice the prior sits in sensory areas without erasing stimulus encoding; in LLaVA the count before the image nearly blinds the model when placed before the image (swing 0.20). We have no neural data of our own.

## 11. Part F, F2a, F2b — the POPE link (Q6)

(Full detail in the separate F report; summarised here with all numbers.)

### 11.1 F — behavioural (510 POPE questions, 170 per split, 323 images; 5 prompts: none, neutral, k2, k5, k8; ≈28 min)
- **Baseline:** random AUROC 0.950, hit 0.776, false-yes 0.012; popular 0.970, 0.765, 0.012; adversarial 0.925, 0.776, 0.082 (the learned prior is visible only on adversarial).
- **Count effect (k8 − k2):** absent: random +0.08, popular +0.07, adversarial +0.05; present: +0.11 / +0.10 / +0.11; **overall +0.09 [0.08, 0.09]** (gratings +0.66). Split differences ≈ 0 (adversarial − random, absent: −0.02 [−0.04, −0.01]).
- **Swing ratio:** 0.79–0.93 (gratings 0.4–0.55); neutral sentence costs about the same.
- **False-yes** none/neutral/k2/k5/k8: random 0.012/0.024/0.035/0.035/0.035; popular 0.012/0.000/0.012/0.012/0.012; adversarial 0.082/0.071/0.118/0.118/0.118. k2, k5, k8 identical.
- **Gain/offset:** a = 0.89 [0.88, 0.90], c = +0.04 (gratings 0.43 / 0.63).
- Not saturation: for uncertain questions (|Δ_none| < 1.5, n = 118) the effect is +0.06. Presence vs count: mean Δ none −0.80, neutral −0.64, k2 −0.37, k5 −0.28, k8 −0.28.

### 11.2 F2a — wordings and the learned-prior proxy (198 questions, 15 prompts, ≈34 min)
Count effect per wording: sentence +0.088 [0.079, 0.096]; list −0.017; hint +0.083; rate +0.074; after +0.200 (present +0.33, absent +0.07, swing 0.58, asymmetric). The explicit hint gives no more. Any yes/no sentence roughly doubles false-yes (0.071 → 0.12–0.15; adversarial 0.18 → 0.27–0.33); k2 = k8; `after` lowers it (0.04–0.05). Blank-image prior proxy P_i: mean −4.39, sd 0.60; weak validity (Spearman +0.09 on absent); count-effect slope on P_i −0.012 per sd; tertiles 0.110 / 0.077 / 0.075; with no evidence (blank image) the effect is +0.161 vs +0.088. Hypothesis, untested: priming by the word "yes".

### 11.3 F2b — mechanism (180 questions; 72 patched, 36 steered; ≈85 min)
- **M1 token patching (gap +0.079):** digits 0.94 at L4 then ≈0.3; sentence words 1.1–1.3 at L8–14; question tokens 0.70/0.86/0.62/0.54 at L16/18/20/24; answer position ≈ 0 (−0.12 … −0.25 at L12–18) until L20, 0.31 at L24; all post-image 1.0. Same first stage as gratings; different late stage (gratings: answer position 0.30/0.73/0.88 at L16/18/20).
- **M2 transfer:** cos(G, P) = −0.847 (L4, p 0.015), −0.390 (L8), −0.134 (L12), −0.106 (L16), −0.069 (L20), −0.073 (L24), −0.053 (L28), all inside the null at L ≥ 12. Early negative = digit-order artefact (first digit is 10−k on gratings, k on POPE). Evidence axes across tasks cos −0.09…+0.15; read-out axes −0.006 → weak test.
- **M2b:** cos(learned-prior axis, POPE count direction) 0.03 / 0.005 / 0.013 / 0.055 / 0.105 / 0.136 (L8…L28), p ≥ 0.15.
- **M3 steering (3× natural, 36 questions):** POPE evidence axis +0.559/+0.580 (L16/L20); gratings count G −0.142/−0.072; POPE count P −0.083/−0.011; learned-prior axis +0.162/+0.188 (residual +0.150, z 1.92 at L16; +0.078, z 2.79 at L20). Count directions do not push "yes"; the learned-prior axis does.

## 12. Answers to Q1–Q6, with caveats

**Q1 — measurable combination.** γ = 0.277 [0.233, 0.322] with counts10; no-history control ≈ 0; screen → full run replicated (0.227 → 0.277); β 15.0 (none) vs 6.3 (history); shifts 5% / 12%. The prior follows the in-prompt count (count-direction decoding AUROC 0.90 = count-only ceiling).

**Q2 — offset or contrast-dependent.** VLM: flat (+0.75). Mouse: no consistent contrast dependence; lapses also move with the prior.

**Q3 — where and how.** Probes (block AUROC 0.87 by L6; side ≈ 1.0 from L2); count direction corr 0.95 from L16; null for "explains shift beyond count"; representation linear, behaviour a step; evidence separation ≈ 0.3–0.4 of no-history from L10 (count-matched, §4c; starts falling from L1); patching: ~0 at L ≤ 12, 0.22/0.53/0.84/~1.0 at L14/16/18/20; steering explained by evidence-axis overlap with a small borderline remainder; token patching digits → words → answer position; prior direction does not cut evidence, history state does.

**Q4 — conflict.** VLM constant offset 0.5–0.85; one gain/offset rule fits both groups; mice 0.93 vs 0.87 (100%), 0.78 vs 0.53 (6%).

**Q5 — mice vs LLaVA.** Shift 9.6% [5.4, 15.0] (mouse) vs 12.2% [8.3, 17.5] (LLaVA, history β; block-bootstrap) or 5.1% [3.5, 6.6] (no-history β). Use history β in the paper. γ is ~2× larger in mice. Intervals are over mice vs over blocks → "not distinguishable". Mice track recent history over the true block (10% block-model wins).

**Q6 — same machinery as POPE's learned prior?** No; separate levers (moderate evidence). Count effect on POPE is ~7–10× smaller than gratings (+0.09 vs +0.66 / 0.86 / 0.907 in main run / B3 / F2b), same on all splits, not a wording artefact; routed through question tokens; not aligned with the learned-prior axis; the learned-prior axis (not the count direction) is a yes-lever.

## 13. Limitations and what is not established

1. One model, 4-bit quantised, one prompt family; the second-model replication (G) was dropped by decision.
2. Coarse mouse comparison: intervals are over different units (mice vs blocks); no neural data of our own.
3. Steering result on gratings is borderline (2 sd, 6 nulls); B2c not run.
4. Digit order is not counterbalanced between tasks (M2 early layers).
5. Evidence and read-out axes differ between the tasks, so a failed transfer is weak evidence.
6. Blank-image learned-prior proxy is noisy (Spearman +0.09 in F2a vs +0.40 in F2b on absent objects).
7. Small mechanistic samples (72 patched, 36 steered; 24 per split).
8. M1 prompt-structure confound (longer POPE suffix; no question-token set in B2).
9. The "yes" priming hypothesis is untested.
10. False-yes counts are small (≈6–7 adversarial/absent questions per sample).
11. Retracted along the way: A2 positive partial correlation (artefact); "suppression begins where the count arrives"; "any extra text costs evidence"; "steering acts through the read-out axis".

**Follow-ups, in priority order:** question-token steering/patching; "yes"-only priming control; digit-order-balanced M2; B2c; B4; second model; confirm Nature-version numbers of Findling et al.

## 14. Background materials and logs used in the project

- `brain-vlm-hallucination-summary.md` (17 Oct 2026 chat summary, UGP-1 context): brain hallucination and predictive coding (strong-prior account, Corlett et al.; conditioned hallucinations, Powers/Mathys/Corlett; Charles Bonnet syndrome as a generative model hallucinating under weak input, Reichert et al.), confabulation as a separate phenomenon, and how each maps to VLMs. It motivates the evidence-vs-prior framing.
- `handoff_confabulation_vs_misinterpretation.md`: handoff note separating confabulation (false content with no grounding) from misinterpretation (wrong use of the evidence) for the hallucination thread.
- `ec0d8ab0-attachment.txt`: chat summary of the dual-pathway paper (arXiv 2605.13156: a visual-grounding pathway vs a hallucination pathway competing).
- Kaggle logs: preflight failure ("dataset slugs and hashlink are all null", fixed by retry + fallback), the B steering cuda:0/cuda:1 `RuntimeError` (fixed by moving vectors to `h.device`), run-time logs of B (1126 s into the run when sampled; "silent for 10 min" was normal between pushes).
- Executed notebooks supplied: main run, QUICK B, full B, B2, D, F, F2a, F2b; output zips: share, A_out, B_out_quick / B_out_final, B2_out, D_out, F_out, F2a_out, F2b_out.

## 15. Files, datasets and reproducibility

| Item | File / dataset |
|---|---|
| Pilots | `pilot2.csv` (P2), stimulus screen and P1 outputs (screenshots); first history analysis cell had the side-composition bug |
| Main run | `prior_vs_evidence_llava.ipynb`; outputs `summary.txt`, `screen_summary.csv`, `probe_results.csv`, `full_hs_*.npy` |
| A | `A_cache_analysis.ipynb` (+ 2 CPU patch cells); `A_summary.txt` |
| B | `B_patching_steering.ipynb`; ckpt `ponishbhatia3/llava-pve-b-ckpt`; `B_summary.txt` |
| B2 | `B2_controls_and_token_patching.ipynb`; `B2_summary.txt` |
| B4 | `B4_conflict_gain_offset.ipynb` (not run) |
| D / D2 | `D_ibl_mouse_behavior.ipynb`, `D2_mouse_lapses.ipynb`; `D_summary.txt`, `mouse_fits.csv`, `conflict_accuracy.csv`, `psychometric_by_block.csv` |
| F / F2a / F2b | `F_pope_inprompt_prior.ipynb`, `F2a_pope_wordings_and_prior.ipynb`, `F2b_pope_mechanism.ipynb`; ckpts `llava-pve-f-ckpt`, `-f2a-ckpt`, `-f2b-ckpt`; `F_report.md` |
| Tracker | https://claude.ai/artifact/2jiyZLZBP81HMGD5mTDeex (v12) |

Kaggle mechanics: checkpoint datasets pushed via the Kaggle API (preflight push retried 3×, then pushing is disabled with a warning); `maybe_push` every 10 min; the resume cell copies files back from an attached checkpoint with the same run config.
Model `llava-hf/llava-1.5-7b-hf` in 4-bit; POPE from `lmms-lab/POPE`; IBL data via ONE / openalyx.
