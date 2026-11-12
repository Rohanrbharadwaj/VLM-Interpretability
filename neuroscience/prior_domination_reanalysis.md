# Prior-domination reanalysis (JEPA-Vicuna vs LLaVA-1.5, POPE object, no new GPU run)

**Source:** `scores.csv` from the encoder-swap comparison: 1000 balanced POPE object questions (500 present, 500 absent), per-question $\Delta=\text{logit(Yes)}-\text{logit(No)}$ for the clean image (`LD_*`) and the noise-corrupted image (`LDC_*`), both models.

**Why:** task-doc Q4 asks whether caption-aligned encoders make embeddings more prone to language-prior domination than an encoder that never saw text. This is a descriptive check with existing data; it does not isolate encoder from training recipe.

## Results

| Measure | LLaVA | JEPA |
|---|---|---|
| Clean AUROC | 0.937 | 0.818 |
| Corrupted-image (prior-only) AUROC | 0.624 | 0.553 |
| Corrupted-image yes-rate | 0.03 | 0.63 |
| Corrupted $\Delta$ mean / sd | −1.50 / 0.89 | +0.18 / 0.55 |
| Clean $\Delta$ mean / sd | +1.05 / 3.39 | −0.13 / 0.61 |
| $R^2$ of clean $\Delta$ on corrupted $\Delta$ | 0.071 | 0.025 |
| Spearman(clean, corrupted), absent objects | 0.25 | 0.09 |
| Spearman(clean, corrupted), present objects | 0.09 | 0.17 |
| False-yes on absent objects, by corrupted-$\Delta$ tercile (low / mid / high) | 0.114 / 0.159 / 0.200 | 0.139 / 0.180 / 0.234 |

Related earlier result (R1 in the final results): at equal hit rate 0.89, false-yes is 0.15 (LLaVA) vs 0.48 (JEPA), difference +0.33 [0.21, 0.42].

## Reading
- JEPA's answers are **not** more tied to the prior-only signal ($R^2$ 0.025 vs 0.071). With the matched-sensitivity result, JEPA's extra hallucination looks like weaker evidence rather than a stronger language-prior pull.
- Both models hallucinate more when the prior-only signal favours "yes" (tercile gradient similar; JEPA slightly higher at every level).
- This does not support "text-free encoder is less prior-dominated"; it also cannot test the caption-alignment hypothesis, because encoder and training recipe differ at once.

## Caveats
- The corrupted conditions are not equivalent across models: LLaVA's corrupted state answers "no" almost always (yes-rate 0.03), JEPA's answers "yes" 63% of the time. Corruption strength was set per model, so the corrupted-AUROC difference should not be over-read.
- Descriptive, one projector and one seed per model, 1000 questions.
- Recipe confound unresolved (LLaVA projector and LM jointly fine-tuned; JEPA projector-only on frozen 4-bit LM). Control A (CLIP stage-1 projector in the same frozen Vicuna) would address it.
