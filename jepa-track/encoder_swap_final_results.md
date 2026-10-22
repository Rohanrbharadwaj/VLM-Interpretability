# Encoder swap on POPE objects: LLaVA-1.5 vs JEPA-Vicuna (results and conclusions)

**Setup.** Same 1000 balanced POPE object questions (265 images), same measurement code. LLaVA-1.5-7B (CLIP ViT-L/14-336, jointly fine-tuned projector and LM, fp16) vs I-JEPA ViT-H/16 → projector trained by us → frozen Vicuna-7B-v1.5 (4-bit). $\Delta=\text{logit(Yes)}-\text{logit(No)}$. Intervals are 95% image-cluster bootstraps. Ablation uses the first 400 questions, patching the first 300 (as in the JEPA runs). Sources: notebook output of the full run, `scores.csv`, `dla_small.npz`, the LLaVA ablation and patching arrays, and the earlier JEPA ablation/patching arrays.

## 0. Bottom line

For object hallucination the JEPA swap does **not** create a different mechanism. It produces **less evidence** for the same mechanism to work with.

- **Same machinery.** Evidence MLPs at layers 16–23 (L18, L20 largest), opposing MLPs (L21/24/27), late braking (L28), the same most necessary window (layers 16–19), the same amount of downstream compensation, and the same route: the image reaches the answer through the K/V of the *text* positions in layers 8–19, not through the image tokens.
- **Different amount of evidence.** The truthful margin is 5.23 logits (LLaVA) vs 0.67 (JEPA); the best linear probe on the last-position residual stream reaches AUROC 0.969 vs 0.887; output AUROC 0.937 vs 0.818. About **69%** of the output gap (0.082 of 0.119) is lower decodable information, **31%** is poorer use of it.
- **Answer to the task question.** Hallucination is *at least partly encoder-side* in the quantitative sense: the JEPA stack delivers less decodable object information, and it hallucinates much more at equal sensitivity. It is *not* qualitatively different, so there is no sign of a JEPA-specific failure mechanism, and a pure "LM binding failure" story is also not supported: in both models about half or more of the missing AUROC is information that is not linearly decodable at all (§3).
- **Scope.** Object existence only. The "encoder" here is the whole upstream stack (encoder + projector + the LM's reading of it); the projector/LM training recipes differ between the two models, so this run cannot separate "I-JEPA features" from "our alignment recipe" (§6).

## 1. Numbers

| | LLaVA-1.5 | JEPA-Vicuna |
|---|---|---|
| AUROC / $d'$ | 0.937 / 2.42 | 0.818 / 1.31 (diff +0.119 [0.095, 0.145]) |
| mean truthful margin $\overline{s\,\Delta}$ | 2.62 | 0.32 |
| hit rate / false-yes at threshold 0 | 0.888 / 0.158 | 0.630 / 0.184 |
| false-yes at hit rate 0.89 | 0.15 | 0.48 (diff +0.33 [0.21, 0.42]) |
| probe ceiling (best layer) | 0.969 (L28) | 0.887 (L26), diff −0.082 [−0.100, −0.064] |
| ceiling − output ("unused") | 0.032 [0.019, 0.044] | 0.068 [0.046, 0.088] |
| noise corruption removes the truthful margin | −2.59 of +2.65 | −0.31 of +0.34 |

## 2. Results

**R1. Behaviour at matched operating points (Fig 1).** At threshold 0 the false-yes rates look alike (0.158 vs 0.184); that is only because JEPA answers "yes" less often (hit rate 0.63 vs 0.89). With the hit rate held equal JEPA hallucinates far more (0.24 vs 0.02 at 0.70; 0.34 vs 0.06 at 0.80; 0.48 vs 0.15 at 0.89). Errors are partly shared: with each model at its best threshold, $P(\text{JEPA wrong}\mid\text{LLaVA wrong})=0.50$ vs $0.24$ when LLaVA is right ($\phi=+0.20$ [0.13, 0.25]; 62 shared of 124 and 269 errors). Most JEPA errors are on questions LLaVA answers correctly, so they are not just hard questions.

**R2. Direct effects (per-question MLP attribution).** The same MLPs carry the truth in both: L18, L20, L23 positive; L21, L24, L27, L28 negative (notebook Fig 2). MLP-level truth parts correlate $\rho=+0.79$; the top-20 components overlap in 15 (0.4 expected by chance); the class-independent constants have the same sign (L20 $+2.5$ / $+3.1$; L28 $-2.6$ / $-3.7$). JEPA's truth parts are far smaller (L18: 0.20 vs 1.56 logits, 7.8×; L20: 0.09 vs 0.61, 6.7×). Over all 1057 components $\rho$ is only $+0.29$, because 1024 noisy head components dominate (JEPA split-half reliability 0.78); heads cannot be compared reliably here. Evidence and opposing MLP sums are strongly anti-correlated within class (−0.97 / −0.92), a cancellation that may partly reflect shared scale or confidence.

**R3. Necessity, measured in margin units (Fig 2).** AUROC is blind to shrinking a margin, so mean-ablation looks harmless for LLaVA ($\Delta$AUROC $-0.002$ for the evidence MLPs, $0.000$ for all second-half MLPs) even though it removes more than half of the truthful margin. In margin units (fraction of $\overline{s\,\Delta}$ retained):

| ablated set | LLaVA | JEPA |
|---|---|---|
| evidence MLPs | 0.41 | 0.31 |
| random same-size sets | 1.02 ± 0.07 | 0.94 ± 0.08 |
| all second-half MLPs | 0.42 | 0.19 |
| all second-half heads | 0.48 | 0.50 |
| opposing MLPs / answer-reactive MLPs | 1.03 / 1.20 | 0.97 / 1.08 |
| most necessary 4-layer MLP window | layers 16–19 (0.45) | layers 16–19 (0.38) |

The profiles over all 129 common configurations correlate at $\rho=+0.80$ (+0.86 without random controls). Evidence MLPs vs random sets: $z=-8.4$ (LLaVA), $-7.7$ (JEPA). The AUROC of JEPA falls because its margin is small relative to the remaining noise; LLaVA's large margin absorbs the same relative loss.

**R4. DLA overstates necessity by about 2×, equally in both (Fig 3).** Predicted effect of replacing the evidence MLPs by their mean (minus their class-dependent attribution) vs the measured effect: slope 0.51 (LLaVA), 0.45 (JEPA). Downstream layers rebuild about half of it. Opposing MLPs are almost fully compensated (0.10, −0.08). This is a first-order estimate (reference mean taken over the evaluation set).

**R5. Route (Fig 4).** Patching the K/V of text positions in layers 8–15 restores 0.52 [0.51, 0.54] (LLaVA) and 0.53 (JEPA) of the corruption effect; the image tokens' own K/V restore $\approx 0$ ($-0.00$ and $-0.16$). Text-minus-image difference in layers 8–15: LLaVA +0.53 [0.51, 0.54], JEPA +0.68 [0.55, 0.88]; layers 16–19 likewise (+0.35, +0.29). Last-position heads+MLPs in layers 8–19 restore 0.6–0.7 in both. The only visible difference is late (layers 28–31), where the last-position components push against the evidence (LLaVA −0.16, JEPA −1.11 with a very wide CI). The old and new clean passes agree exactly on the 300 questions (max |difference| 0.0), so the comparison is on identical inputs.

**R6. Information vs use (Fig 5).** Probes on the last-position residual stream reach their ceiling by layer ~10 in LLaVA and ~14 in JEPA. AUROC budget (perfect = 1):

| | output | present but unused | not linearly decodable |
|---|---|---|---|
| LLaVA | 0.937 | 0.032 | 0.031 |
| JEPA | 0.818 | 0.068 | 0.113 |

So 49% of LLaVA's and 62% of JEPA's shortfall from perfect is information that a linear readout of the same state cannot recover (the rest is present but unused). The probe gets 59% of the errors right in both models (no difference), which says that roughly 60% of errors in each are readout/use problems of information that is present.

## 3. The pre-registered criteria, as run and as corrected

| criterion | as run | corrected reading |
|---|---|---|
| M1 same most necessary window / window profile | FAIL (AUROC: LLaVA 28–31 vs JEPA 16–19, $\rho=-0.38$) | PASS in margin units (both 16–19; 8-window $\rho=+0.76$) |
| M2 evidence MLPs beat random sets in LLaVA | FAIL ($z=-0.8$) | PASS in margin units ($z=-8.4$) |
| M3 text-token K/V route in both | not run in the notebook (patching file not attached) | PASS (R5) |
| M4 truth parts correlate $\ge0.6$, constants same sign | FAIL ($\rho=0.33$ corrected) | PASS for MLPs ($\rho=0.79$, constants agree); heads unresolvable |
| M5 evidence/opposing anti-correlated in both | PASS | PASS |
| I1 JEPA ceiling lower | PASS | PASS |
| I2 both leave information unused (lower CI bound > 0.02) | FAIL (LLaVA bound 0.019) | borderline; JEPA clearly yes, LLaVA 0.032 |
| I3 probe recovers more of JEPA's errors | FAIL | no difference (0.59 vs 0.59) |
| I4 LLaVA better output AUROC | PASS | PASS |

The notebook's automatic verdict, "1 of 4 criteria, DIFFERENT mechanism", is the pre-registered result and I would report it as such. M1, M2 and M4 were specified in AUROC (rank) terms, which cannot see a margin that shrinks but keeps its ordering; the margin-based rows are a **post hoc** correction motivated by that, not a pre-specified test. Under the earlier rule in `final_findings.md`, criteria (i) shifted evidence layers and (ii) a direct image-token role are **not** met, while (iii) (JEPA's evidence-MLP AUROC drop lies outside LLaVA's random range) **is** met in AUROC form and not in margin form. Which form to treat as binding is a judgement call; the margin form is the better-posed measurement, but report both.

## 4. Answers to the original questions

- **Q1. Does the answer depend on the image?** Yes in both, to different degrees. Corrupting the image removes the entire truthful margin (R5/§1). The corrupted runs are not at chance (AUROC 0.62 LLaVA, 0.55 JEPA), so some information survives the noise, and corrupted LLaVA collapses to answering "no" (yes-rate 3%).
- **Q2. Where is the evidence turned into the answer?** Evidence MLPs in layers 16–23 at the last position (L18, L20 largest), opposed by L21/24/27 and braked at L28; the same layers in both models (R2, R3).
- **Q3. Is the localisation causal and specific?** Yes: removing the five evidence MLPs removes 59% (LLaVA) / 69% (JEPA) of the margin, random same-size sets almost nothing. But only about half of the direct effect reaches the output in both models (R4), so attribution alone overstates it.
- **Q4. How does the information travel?** Through the K/V of the text positions in layers 8–19; image-token K/V play no direct role in either model (R5).
- **Q5. What separates correct from wrong answers?** For both models about 60% of errors are cases where a linear probe on the same state is right; JEPA's extra errors are mostly on questions LLaVA handles (R1, R6). Descriptive, not a mechanism.
- **Q6. Encoder vs language model?** See §0: same LM-side mechanism, weaker and less decodable evidence under JEPA; the quantitative gap is mostly information (69%) and the rest use (31%).

## 5. Results that likely generalise

1. **Judge necessity by margin, not AUROC.** Rank metrics saturate for confident models and make ablations look harmless (here: all second-half MLPs removed, AUROC unchanged, margin −58%).
2. **Direct logit attribution overestimates necessity by about 2×**, because later layers rebuild part of the signal; the compensation was the same in two very different VLMs.
3. **The yes/no evidence circuit in Vicuna-derived VLMs is shared across vision encoders** (CLIP with fine-tuned LM vs frozen-LM I-JEPA): same layers, same route, same cancellation structure.
4. **Compare hallucination rates at matched sensitivity.** At a fixed threshold the models look alike (0.158 vs 0.184) while JEPA is far worse at equal hit rate.

## 6. What is not established

- **Which part of the upstream stack is responsible.** LLaVA's projector and LM were fine-tuned jointly; ours is a projector-only alignment on a frozen 4-bit LM. The lower information ceiling can come from I-JEPA's features, our projector, or the alignment data and recipe. Isolating it needs a CLIP encoder trained with the same recipe, or a JEPA encoder in LLaVA's recipe.
- **Attribute and relation hallucination**, which the task document targets. Nothing here covers them.
- **Heads.** Per-head DLA and patching comparisons are noisy for JEPA; head-level equivalence is untested.
- **The 69/31 split** depends on the probe (supervised, linear, best layer) and on AUROC as the scale.
- **Noise corruption:** LLaVA's calibrated $k=1.0$ is not at chance (0.62 on the full set) and drives it to "no"; the signed patching fractions are defined relative to the truthful margin, which the corruption removes completely in both models, but the corrupted state is off-distribution.
- **Single question set, single run;** the first-order compensation estimate uses the evaluation-set mean as the reference.

## 7. Corrections to earlier statements

- "False-yes rates are similar, so JEPA's deficit is mostly misses" was a threshold-0 artefact; at equal sensitivity JEPA hallucinates much more (R1).
- The earlier JEPA-only verdict "leaning LM-side" is revised: with the LLaVA reference the information ceiling is the larger part of the gap.
- "LLaVA evidence MLPs are not necessary" (AUROC) is wrong in margin terms (R3).

## 8. Next steps

1. Optional follow-up notebook already built: combined MLP+heads ablation in both models (tests the redundancy explanation directly) and a rerun of the analysis; it should also report margin retention (to be added).
2. Attribute/relation extension (`HALLUC_TYPE`), using Rohan's data for LLaVA.
3. A like-for-like control (CLIP encoder with the same projector-only recipe) if the encoder-vs-recipe split matters for the write-up.
