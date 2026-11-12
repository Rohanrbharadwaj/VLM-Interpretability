# Does an in-context prior use the same machinery as the learned prior behind POPE hallucination?

Report on notebooks **F**, **F2a** and **F2b** (LLaVA-1.5-7B, 4-bit, 2×T4). Q6 of the project checklist.

---

## 1. Bottom line

- **Behaviour.** Telling LLaVA "in the last 10 questions the answer was yes $k$ times" barely moves its POPE answers: **+0.09 logits** for $k{=}8$ vs $k{=}2$, against **+0.66** on the Gabor-grating task. The effect is the same on the random, popular and adversarial splits (no interaction with the learned prior) and is not a wording artefact (five wordings tested; none above +0.2, three at ≈+0.08, one negative).
- **What *does* move POPE** is the mere presence of a yes/no history sentence, whatever the count: it raises $\Delta=\text{logit(Yes)}-\text{logit(No)}$ by ≈+0.4–0.5 and roughly doubles the false-yes rate. A neutral sentence does nothing. I read this as token priming, not as a use of a base rate (not tested directly).
- **Mechanism.** The count and the learned prior are separate: (i) the gratings count direction does not transfer to POPE beyond the null at layers ≥12 (a weak test, see §6); (ii) the image-free learned-prior axis is not aligned with the POPE count direction (|cos| ≤ 0.14, all $p\ge0.15$); (iii) adding the POPE count direction at the answer position does **not** push "Yes", while adding the learned-prior axis does (+0.08 to +0.15 logits beyond what its overlap with the evidence axis predicts); (iv) the count is read through the question tokens at layers 16–24, and the answer position only gets it at ≈L24, unlike the gratings where the answer position took over at L16–20.
- **Answer to Q6:** on LLaVA-1.5-7B the in-context prior (count) and the learned prior behave as **separate levers**, both behaviourally and mechanistically. The evidence is moderate (one model, one prompt family, 4-bit, small samples for the mechanistic tests).

---

## 2. The question

The project compares how LLaVA combines **evidence** (a Gabor grating, left or right) with a **prior** (trial-history counts in the prompt, an analogue of the IBL mouse block prior) and asks whether this resembles mice and connects to POPE-style hallucination. Q6 is the link to hallucination:

> **Q6. Does the in-context prior use the same machinery as the learned prior behind POPE?**

POPE hallucination (answering "yes" to absent objects) comes from a prior *learned in training* (object co-occurrence; strongest on the **adversarial** split). The in-context count is a prior supplied *at inference*. The project's earlier checklist items were:

- q6a: steer the prior direction on POPE splits (margin change vs a random direction)
- q6b: patch on POPE splits; compare necessity
- q6c: repeat on a second model (dropped by the user)

After notebook B2 showed that steering directions on the gratings are mostly explained by their overlap with the *evidence* axis (no clean count-specific direction; borderline remainder +0.13 to +0.17 logits at ≈2 sd), a purely mechanistic "steer the prior direction" test was judged unjustified. So F was run **behavioural only** first, and F2a/F2b were added afterwards to answer the wording and mechanism questions in a controlled way.

**Reference values from the gratings (for comparison throughout).** Count effect $\Delta(k{=}8)-\Delta(k{=}2)$ ≈ +0.66 logits; rule $\Delta_{hist}\approx a\,\Delta_{none}+c\,\mathrm{sign}(\text{count})+b$ with gain $a\approx0.43$ and offset $c\approx0.63$; swing ratio (evidence retained) ≈ 0.4–0.55; token-position patching (B2): digits matter at L4, sentence words at L8–14, the answer position at L16–20 (0.30/0.73/0.88 at L16/18/20), all post-image positions give 1.0.

---

## 3. Notebook F: behaviour on POPE

### Design
510 POPE questions from `lmms-lab/POPE` (170 per split: random, popular, adversarial; half present, half absent; 323 distinct images). Five prompts per question, one forward pass each, no hooks:

| Prompt | Text inserted after `<image>` |
|---|---|
| none | – |
| neutral | "The picture was taken with a plain camera in a quiet room on a sunny afternoon." (length control) |
| k2, k5, k8 | "In the last 10 questions the answer was yes $k$ times and no $10-k$ times." |

$\Delta=\text{logit(Yes)}-\text{logit(No)}$ at the first answer token. Confidence intervals come from an image-clustered bootstrap (500 draws).

**Questions asked:** (1) does the count shift $\Delta$, and by how much? (2) is it the same on the three splits and on present/absent objects (additive, interacting or competing with the learned prior)? (3) what does it cost the evidence (swing/margin)? (4) does it change the false-yes rate? (5) does the gratings gain/offset rule carry over?

### F1: baseline (no history)

| Split | AUROC | Hit rate | False-yes | Truthful margin | Mean $\Delta$ (absent) |
|---|---|---|---|---|---|
| random | 0.950 | 0.776 | 0.012 | 2.24 | −3.07 |
| popular | 0.970 | 0.765 | 0.012 | 2.33 | −3.34 |
| adversarial | 0.925 | 0.776 | 0.082 | 2.04 | −2.60 |

The learned prior is visible only on the adversarial split (false-yes 8% vs 1%; ≈7 of 85 absent questions). LLaVA leans "No" under this prompt (hit rate ≈0.77).

### F2/F3: the count effect, $\Delta(k{=}8)-\Delta(k{=}2)$ [95% CI]

| Split | Absent objects | Present objects |
|---|---|---|
| random | +0.08 [0.07, 0.09] | +0.11 [0.09, 0.12] |
| popular | +0.07 [0.05, 0.08] | +0.10 [0.09, 0.11] |
| adversarial | +0.05 [0.04, 0.06] | +0.11 [0.09, 0.12] |
| **all** | **+0.09 [0.08, 0.09]** | |

Split differences: absent objects, popular−random −0.01 [−0.03, +0.01], adversarial−random −0.02 [−0.04, −0.01]; present objects ≈0. The effect is ≈7× smaller than on the gratings and **does not grow where the learned prior is strong** (if anything it is marginally smaller on adversarial/absent). Additive-and-separate, not interacting.

### F4: evidence cost (swing ratio; 1 = no loss)

| Split | neutral | k2 | k5 | k8 |
|---|---|---|---|---|
| random | 0.90 | 0.92 | 0.91 | 0.93 |
| popular | 0.79 | 0.89 | 0.88 | 0.89 |
| adversarial | 0.83 | 0.88 | 0.87 | 0.89 |

On the gratings the count sentence cost about half the evidence (0.4–0.55). Here the history sentence costs ≈10%, and the *neutral* sentence costs about as much.

### F5: hallucination and hit rates

| Split | none | neutral | k2 | k5 | k8 |
|---|---|---|---|---|---|
| false-yes, random | 0.012 | 0.024 | 0.035 | 0.035 | 0.035 |
| false-yes, popular | 0.012 | 0.000 | 0.012 | 0.012 | 0.012 |
| false-yes, adversarial | 0.082 | 0.071 | 0.118 | 0.118 | 0.118 |
| hit rate, random | 0.776 | 0.788 | 0.847 | 0.847 | 0.835 |
| hit rate, popular | 0.765 | 0.753 | 0.800 | 0.800 | 0.800 |
| hit rate, adversarial | 0.776 | 0.788 | 0.824 | 0.824 | 0.824 |

False-yes and hit rates are **identical for k2, k5 and k8**: any yes/no history raises both, the count does not matter.

### F6: gain/offset rule per split

| Split | Gain $a$ | Offset $c$ |
|---|---|---|
| random | 0.90 [0.88, 0.92] | +0.05 |
| popular | 0.88 [0.86, 0.90] | +0.04 |
| adversarial | 0.90 [0.88, 0.92] | +0.04 |
| all | 0.89 [0.88, 0.90] | +0.04 [0.04, 0.05] |
| *gratings* | *0.43* | *0.63* |

The gratings rule (lose half the evidence, gain a count offset) does not carry over: the model keeps ≈90% of its evidence gain and the offset is ≈0.04.

### Extra checks on F's raw rows
- **Not saturation.** Even on uncertain questions ($|\Delta_{none}|<1.5$, $n=118$) the count effect is only +0.06 (fitted $a=1.00$, $c=0.03$). The model ignores the count even when the answer is close to a coin flip.
- **Presence vs count.** Mean $\Delta$: none −0.80, neutral −0.64, k2 −0.37, k5 −0.28, k8 −0.28. Most of the shift appears when *any* yes/no sentence is present; the count adds little ($k5-k2=+0.08$, $k8-k5=+0.003$).

### Reading
Same small shift on all splits and for present/absent objects ⇒ the in-context count does not interact with the learned prior at the level of behaviour. LLaVA answers POPE from the image and mostly ignores the count.

---

## 4. Notebook F2a: wordings and the per-question learned prior

### Design
198 new POPE questions (66 per split, seed 3, 166 images), 15 prompts each: none, neutral, five wordings × $k\in\{2,8\}$, and three blank-gray-image prompts (no history, sentence $k2$, sentence $k8$). Wordings: **sentence** (as in F); **list** ("Answers to the last 10 questions: yes, yes, no, …"); **hint** (sentence + "Most of the time the answer is yes/no."); **rate** ("In this dataset, 80%/20% of the answers are yes."); **after** (the sentence placed after the question).

**Questions asked:** is the weak effect specific to one wording? Does the count effect depend on how strong the learned prior is for each *question* (finer than the three splits)? The learned prior is proxied by $P_i$ = $\Delta$ on a blank image (what LLaVA expects for that object question with no evidence).

### Baseline in this sample
AUROC 0.939; hit rate 0.798; false-yes 0.071 (adversarial 0.182; ≈6 questions, so noisy).

### A1: count effect per wording [95% CI]

| Wording | Count effect (k8−k2) | Absent | Present | Swing ratio | Presence shift |
|---|---|---|---|---|---|
| sentence | +0.088 [0.079, 0.096] | +0.064 | +0.111 | 0.93 | +0.49 |
| list | −0.017 [−0.027, −0.007] | −0.037 | +0.004 | 0.92 | +0.44 |
| hint | +0.083 [0.073, 0.095] | +0.064 | +0.102 | 0.91 | +0.41 |
| rate | +0.074 [0.059, 0.087] | +0.062 | +0.086 | 0.76 | +0.54 |
| after | +0.200 [0.176, 0.225] | +0.074 | +0.327 | 0.58 | +0.08 |

(Neutral sentence: presence shift +0.083, swing ratio 0.89. "Presence shift" = mean over $k=2,8$ of $\Delta$ minus $\Delta_{none}$.)

- The sentence wording **replicates F** (+0.088 vs +0.09) on a fresh sample.
- Even the explicit **hint** ("most of the time the answer is yes") gives no larger effect: the model does not use the count as information.
- **after** is the one wording that behaves somewhat like the gratings (swing ratio 0.58, gain 0.58, offset +0.10), but the effect is asymmetric (+0.33 on present objects, +0.07 on absent) and so is not a clean symmetric prior shift. It also lowers the false-yes rate (0.07 → 0.04–0.05).
- No wording shows a split difference in any consistent direction (A1c; all split-level gaps are within ≈0.03 of each other except `after` on present objects, which is the same across splits).

### A1b: yes/no history sentences raise hallucination regardless of the count

| Prompt | False-yes (all) | False-yes (adversarial) | Hit rate |
|---|---|---|---|
| none | 0.071 | 0.182 | 0.798 |
| neutral | 0.071 | 0.182 | 0.808 |
| sentence k2 / k8 | 0.141 / 0.152 | 0.333 / 0.333 | 0.869 / 0.869 |
| list k2 / k8 | 0.152 / 0.141 | 0.303 / 0.303 | 0.869 / 0.859 |
| hint k2 / k8 | 0.121 / 0.121 | 0.273 / 0.273 | 0.859 / 0.869 |
| rate k2 / k8 | 0.131 / 0.141 | 0.303 / 0.303 | 0.869 / 0.869 |
| after k2 / k8 | 0.040 / 0.051 | 0.121 / 0.121 | 0.768 / 0.788 |

Having a history sentence roughly doubles the false-yes rate; $k=2$ and $k=8$ are indistinguishable. The "rate" wording, which contains only the word "yes", has the largest presence shift (+0.54). My hypothesis is lexical priming by the token "yes" in the context; this was **not** tested (see §7).

### A2: the learned-prior proxy per question
- The blank-image answers barely vary: $P_i$ mean −4.39, sd 0.60, range −5.50 to −3.22.
- **Validity of the proxy is weak in this sample:** Spearman($P_i$, real-image $\Delta$) +0.09 on absent objects and −0.34 on present objects; AUROC of $P_i$ for "model said yes with the real image" among absent objects 0.59.
- Count effect regressed on standardised $P_i$ and $E_i$ ($E_i$ = real-image $\Delta$ without history): slope on $P_i$ −0.012 [−0.020, −0.005]; slope on $E_i$ +0.025 [+0.016, +0.033]. By tertile of $P_i$: low 0.110, mid 0.077, high 0.075. The effect is slightly *larger* where the model expects "no", the opposite of an amplified prior, and tiny. Given the weak proxy I do not interpret this further.
- **No evidence (blank image):** count effect +0.161 vs +0.088 on real images; per-question Spearman −0.09 (no relation). The count acts about twice as strongly when there is no evidence, which is the gain-structure direction but is still small.
- Gain/offset on real images: sentence $a=0.90, c=+0.044$; list 0.90, −0.008; hint 0.89, +0.042; rate 0.74, +0.037; after 0.58, +0.100.

---

## 5. Notebook F2b: mechanism

### Design
180 POPE questions (60 per split, seed 0) with stored last-position states at layers {4, 8, 10, 12, 14, 16, 18, 20, 22, 24, 28} for: no history, $k2$, $k8$, blank image; plus 240 gratings (contrast 0.125 for the count direction, contrast 1.0 for the evidence axis). Four tests:

- **M1, token-position patching** of the POPE count effect (72 questions; layers 4–24): overwrite a set of positions in the $k2$ run with the $k8$ run's states. Recovery = fraction of the gap that moves. Position sets: digits, sentence words, question tokens ("suffix"), answer position, all post-image positions.
- **M2, representation transfer.** Cosine and projection between the gratings count direction $G$ (paired $k8-k2$ states) and the POPE count direction $P$, with sign-flip nulls; also with the evidence axes projected out. **M2b:** cosine between $P$ and an image-free *learned-prior axis* (mean blank-image state of the top third minus the bottom third of blank-image $\Delta$), with a permutation null.
- **M3, steering transfer** (36 questions; layers 16 and 20; size 3× the natural POPE count shift; both signs): add $G$, $P$, the learned-prior axis (each also with the evidence axes projected out), and B2's controls: the POPE evidence axis, 3 calibration directions at fixed cosine with it, and 6 sign-flip nulls. Residual = effect minus slope × cosine with the evidence axis.

### Self-tests and scale
All self-tests passed (layer indexing exact, position bookkeeping, steering sign). Natural count effect on POPE in this sample: +0.086 (replicates F); gratings count effect at contrast 0.125: +0.907. Direction reliabilities (split-half cosine) are ≥0.99 for both tasks, so directions are not noise.

### M1: where the count is read (recovery [95% CI]; 72 questions; gap +0.079, sd 0.060)

| Layer | Digits | Sentence words | Question tokens | Answer position | All post-image |
|---|---|---|---|---|---|
| 4 | 0.94 [0.84, 1.07] | 0.73 [0.62, 0.89] | −0.02 | −0.01 | 1.0 |
| 8 | 0.28 [0.23, 0.34] | 1.20 [1.11, 1.29] | −0.02 | 0.00 | 1.0 |
| 12 | 0.31 [0.24, 0.38] | 1.29 [1.19, 1.42] | −0.10 | −0.12 [−0.18, −0.07] | 1.0 |
| 14 | 0.20 [0.15, 0.25] | 1.10 [0.98, 1.25] | 0.02 | −0.25 [−0.36, −0.17] | 1.0 |
| 16 | −0.01 | 0.43 [0.36, 0.51] | 0.70 [0.64, 0.78] | −0.22 [−0.35, −0.09] | 1.0 |
| 18 | 0.01 | 0.28 [0.21, 0.37] | 0.86 [0.76, 0.99] | −0.20 [−0.38, −0.05] | 1.0 |
| 20 | −0.02 | 0.27 [0.21, 0.34] | 0.62 [0.55, 0.71] | 0.03 [−0.13, 0.12] | 1.0 |
| 24 | 0.00 | 0.13 [0.08, 0.19] | 0.54 [0.48, 0.61] | 0.31 [0.21, 0.38] | 1.0 |

- **Same first stage as on the gratings:** digits at L4, sentence words at L8–14 (recovery ≥1: overwriting the words alone over-corrects, so the other positions partly counteract).
- **Different late stage:** the count moves from the sentence words into the **question tokens at L16–24**; the answer position carries almost none of it at L16–20 (≈0 or slightly negative) and only 0.31 at L24. On the gratings the answer position took over at L16–20 (0.30/0.73/0.88). By split (24 questions each) the pattern is the same: sentence words at L12: 1.20/1.25/1.49 (random/popular/adversarial); answer position at L20: 0.18/0.05/−0.24.
- **Caveat:** B2 had no question-token set, and the POPE prompt has a longer suffix (25 tokens, including the "answer with a single word or phrase" instruction) than the gratings prompt. Part of the difference could be prompt structure.

### M2: do the two count directions coincide?

| Layer | cos($G$,$P$) | Null 95% of \|cos\| | $p$ | \|P\|/\|G\| |
|---|---|---|---|---|
| 4 | −0.847 | 0.843 | 0.015 | 1.03 |
| 8 | −0.390 | 0.405 | 0.238 | 1.22 |
| 12 | −0.134 | 0.140 | 0.355 | 0.57 |
| 16 | −0.106 | 0.121 | 0.392 | 0.33 |
| 20 | −0.069 | 0.109 | 0.442 | 0.24 |
| 24 | −0.073 | 0.098 | 0.422 | 0.31 |
| 28 | −0.053 | 0.067 | 0.422 | 0.29 |

(With the evidence axes projected out the numbers are almost identical.)

- At **layers ≥12** the alignment is inside the null ($p\approx0.35$–$0.44$).
- At **L4–10** there is a large *negative* alignment. I attribute it to a **digit-order artefact**, not a shared concept: in the gratings prompt the first digit is $10-k$ ("left 2 times and right 8"), in the POPE prompt it is $k$ ("yes 8 times and no 2"), so the early difference vectors point in opposite directions. A shared "majority label" concept would give a *positive* cosine. The design was not counterbalanced for this.
- **Calibration (M2c):** the two evidence axes (right-vs-left grating; object present-vs-absent) are essentially orthogonal across tasks (cos −0.09 to +0.15), and so are the two read-out axes (cos −0.006). A failed transfer is therefore **weak evidence** about shared machinery: the two tasks read out through different axes.
- The gratings count direction leans heavily on its own evidence axis at late layers (cos 0.64–0.74 at L20–28, as in B2); the POPE count direction is almost orthogonal to the POPE evidence axis (cos −0.03 to +0.21), i.e. a separate direction from evidence.

### M2b: learned-prior axis vs the POPE count direction

| Layer | cos(prior, $P$) | $p$ | cos(prior, evidence axis) |
|---|---|---|---|
| 8 | 0.028 | 0.742 | −0.451 |
| 12 | 0.005 | 0.898 | −0.274 |
| 16 | 0.013 | 0.780 | 0.019 |
| 20 | 0.055 | 0.450 | 0.187 |
| 24 | 0.105 | 0.232 | 0.277 |
| 28 | 0.136 | 0.150 | 0.274 |

The two are not aligned at any layer. Validity of the image-free prior proxy in this sample: Spearman(blank $\Delta$, real-image $\Delta$) is +0.40 on absent and −0.17 on present objects. The real-image state projected on the prior axis correlates strongly with the real-image answer on absent objects (+0.70/+0.87/+0.92 at L16/20/24), which is expected for an axis that is yes-related at layers near the read-out, so I treat it as a sanity check only.

### M3: steering at the answer position (effect = mean of $[\Delta(+d)-\Delta(-d)]/2$, logits)

| Direction | L16 effect (residual, $z$) | L20 effect (residual, $z$) |
|---|---|---|
| POPE evidence axis | 0.559 (−0.020, −0.26) | 0.580 (−0.009, −0.31) |
| gratings count $G$ | −0.142 (−0.092, −1.18) | −0.072 (−0.029, −1.03) |
| $G$, evidence projected out | −0.072 (−0.072, −0.92) | −0.017 (−0.017, −0.61) |
| POPE count $P$ | −0.083 (−0.064, −0.82) | −0.011 (−0.021, −0.77) |
| $P$, evidence projected out | −0.065 (−0.065, −0.83) | −0.020 (−0.020, −0.72) |
| **learned-prior axis** | **+0.162 (+0.150, +1.92)** | **+0.188 (+0.078, +2.79)** |
| prior axis, evidence projected out | +0.152 (+0.152, +1.94) | +0.081 (+0.081, +2.90) |

Slope (effect per unit cosine with the evidence axis): 0.579 (L16), 0.589 (L20); null/calibration residual sd 0.078 (L16), 0.028 (L20). Natural count effect on these questions: +0.087, so a direction as strong as the natural count would give about +0.26 at 3× size if fully effective.

- The POPE **count direction itself does not push "Yes"** at the answer position (effects −0.01 to −0.08), and neither does the gratings count direction: both residuals are inside the null spread. This agrees with M1 (answer-position patching recovers ≈0 at L16–20).
- The image-free **learned-prior axis does push "Yes"**, with a residual of +0.08 (L20, $z$≈2.8) and +0.15 (L16, $z$≈1.9) beyond what its overlap with the evidence axis predicts. This is moderate evidence (36 questions, two layers) and the axis was built from yes-tendency, so it is expected to be a yes-lever; the point is that the count direction is not.

---

## 6. Synthesis: the questions and their answers

| Question | Answer | Evidence |
|---|---|---|
| Does the in-context count shift POPE answers, and by how much? | Barely: +0.09 logits ($k8-k2$), vs +0.66 on gratings | F2; replicated in F2a and F2b |
| Same on the three splits and present/absent (additive, interacting, competing)? | Same everywhere; no interaction with the learned prior (adversarial/absent marginally smaller, −0.02) | F2/F3; F2a A1c |
| Evidence cost (the gratings' gain loss)? | ≈10% (swing ≈0.9, gain ≈0.89, offset ≈0.04), vs ≈50% on gratings; the neutral sentence costs about as much. Exception: `after` wording (swing 0.58, asymmetric) | F4, F6; F2a A1 |
| Does the count change the hallucination rate? | A yes/no history sentence ≈doubles false-yes, but the count ($k2$ vs $k8$) does not matter | F5; F2a A1b |
| Is the weak effect a wording artefact? | No: five wordings, none above +0.2; even an explicit hint gives +0.08 | F2a A1 |
| Does it depend on the question's learned prior? | No clear dependence (proxy weak; slight opposite-sign slope) | F2a A2 |
| Same routing as on the gratings? | Early stage yes (digits → sentence words, L4–14); late stage differs (question tokens at L16–24, answer position only ≈L24) | F2b M1 |
| Shared count representation across tasks? | Not beyond the null at L≥12; weak test (evidence and read-out axes themselves are unaligned; early negative alignment is a digit-order artefact) | F2b M2 |
| Count direction aligned with the learned-prior axis? | No (|cos| ≤ 0.14, $p\ge0.15$) | F2b M2b |
| Does adding each direction move the answer? | Learned-prior axis: yes (+0.08 to +0.15 beyond evidence overlap); count directions: no | F2b M3 |
| **Q6: same machinery?** | **No — separate levers (moderate evidence)** | all of the above |

**One-paragraph conclusion.** On the gratings, where the evidence is ambiguous, LLaVA uses the in-context count: it halves its evidence gain, adds a count-driven offset, and reads the count at the answer position through layers 16–20. On POPE the evidence is strong, the count adds almost nothing beyond the presence of a yes/no sentence, and what little it adds is carried through the question tokens rather than the answer position. The learned co-occurrence prior behind POPE hallucination is a different axis that does act at the answer position. So the in-context prior and the learned prior are not the same lever in this model.

---

## 7. Limitations and what would strengthen the claim

1. **One model, 4-bit quantised, one prompt family.** The second-model replication (q6c) was dropped.
2. **Digit order is not counterbalanced between tasks** (M2). The early-layer negative alignment is probably an artefact; a balanced design would remove the ambiguity.
3. **Transfer tests are weak by construction:** gratings answer with Right/Left and POPE with Yes/No; the evidence axes and read-out axes are orthogonal across tasks.
4. **The image-free learned-prior proxy is noisy** (blank-image $\Delta$ barely varies; validity differs between samples: Spearman +0.09 in F2a vs +0.40 in F2b on absent objects).
5. **Small samples for the mechanistic tests** (72 questions for M1, 36 for M3; 24 per split in the split breakdown). The learned-prior steering residual is $z\approx2$–3 at two layers, not conclusive on its own.
6. **M1 prompt-structure confound:** the POPE suffix is longer than the gratings suffix, and B2 had no question-token set.
7. **The priming interpretation** (the word "yes" in the context raises $\Delta$, independent of the count) is a hypothesis, not a tested result.
8. False-yes rates rest on few questions (adversarial/absent ≈6–7 per sample).

**Follow-ups, in priority order:**
- A steering/patching test at the **question-token positions** (where M1 says the count sits), the fair causal test for the count direction.
- A **priming control**: sentences containing only "yes", only "no", or the digits without the count.
- A **digit-order-balanced** version of M2.
- A second model (if the claim is to be general).

---

## 8. Files and reproducibility

| Notebook | Role | Questions | Seed | Run time | Checkpoint dataset |
|---|---|---|---|---|---|
| `F_pope_inprompt_prior.ipynb` | behaviour, 5 prompts | 510 (170/split) | 0 | ≈28 min | `llava-pve-f-ckpt` |
| `F2a_pope_wordings_and_prior.ipynb` | 5 wordings + blank-image prior | 198 (66/split) | 3 | ≈34 min | `llava-pve-f2a-ckpt` |
| `F2b_pope_mechanism.ipynb` | M1–M3 | 180 (60/split); 72 patched, 36 steered | 0 | ≈85 min | `llava-pve-f2b-ckpt` |

Model `llava-hf/llava-1.5-7b-hf` in 4-bit; POPE from `lmms-lab/POPE` (9000 rows, splits random/popular/adversarial); $\Delta$ is the first-token logit difference Yes−No ("▁Yes"/"▁No"); bootstraps cluster by image. Outputs: `F_summary.txt`, `F2a_summary.txt`, `F2b_summary.txt`, with the CSVs (`pope_rows.csv`, `wording_rows.csv`, `m1_tokpatch_summary.csv`, `m2_alignment.csv`, `m3_steering.csv`, `tokpatch_rows.csv`, `steer_rows.csv`) and overview figures.
