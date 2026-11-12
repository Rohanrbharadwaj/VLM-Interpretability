# Redundancy follow-up: combined MLP + head mean-ablation (last position, second half = layers 16-31), 400 questions, 43 configs/question, random layer-set controls

Run: notebook encoder-swap follow-up (outputs in Kaggle dataset ponishbhatia/encoder-swap-followup). patching_sweeps.npz was NOT attached, so D5 (K/V route) and criterion M3 were not evaluated in this run.

## Margin retained (primary metric: mean(sign(truth)*logit diff) relative to unablated)
| model | MLPs alone | heads alone | together | additive expectation (1 - sum of losses) | loss beyond sum |
|---|---|---|---|---|---|
| LLaVA | 0.42 | 0.48 | 0.13 [0.09,0.17] | -0.10 | -0.23 (overlap) |
| JEPA  | 0.19 | 0.50 | 0.07 [0.05,0.10] | -0.31 | -0.39 (overlap) |
- Pathways partially overlap (combined removal is smaller than the sum) but are not fully redundant: ablating both removes ~87% (LLaVA) / ~93% (JEPA) of the margin.
- AUROC change, second-half MLPs+heads together: LLaVA -0.002, JEPA -0.115 (JEPA MLPs alone -0.202, heads alone +0.024). LLaVA's ranking survives even when the margin collapses; JEPA's does not.
- Sanity: ablating every MLP and head at the last position -> margin 0.02 (LLaVA) / -0.01 (JEPA), AUROC 0.500 in both. Passes.

## Same circuit, now including heads
- Most necessary 4-layer window: MLPs+heads layers 16-19 in both models; 8-layer window: layers 16-23 in both.
- Spearman of retained margin, LLaVA vs JEPA: +0.98 across 8 four-layer windows and +0.98 across all 22 named configs.
- evidence MLPs + evidence heads, margin retained: LLaVA 0.323, JEPA 0.197.

## Re-run verdict table (from the same run)
- MECHANISM: 1 of 4 criteria met -> "DIFFERENT mechanism" (auto verdict). M4 (direct-effect signature) FAILED: rho +0.29 (corrected 0.33), same-sign L20/L28 constants. M5 (evidence/opposing MLP anti-correlation, compensation) PASSED: LLaVA -0.97/-0.98, JEPA -0.92/-0.92. M1-M3 lines not inspected here; M3 needs patching_sweeps.npz.
- I1 PASS: JEPA probe ceiling lower than LLaVA's (-0.082 [-0.100,-0.064]).
- I2 FAIL (borderline): unused information LLaVA +0.032 (0.019,0.044), JEPA +0.068 (0.046,0.088); lower bound for LLaVA below the 0.02 threshold.
- I3 FAIL: share of model errors the probe gets right is equal, JEPA 0.59 (0.53,0.65) vs LLaVA 0.59 (0.51,0.68). No extra LM-side error recovery in JEPA.
- I4 PASS: AUROC LLaVA - JEPA +0.119 (0.095,0.145), d' 2.42 vs 1.31.

## Reading (for the report)
- Location and necessity profile across depth agree between the models (rho 0.98 incl. heads); the per-component direct-effect signature does not (rho ~0.3). Same layers, different component-level detail. State it as "same layers, partly different components", not as "same circuit" or "different mechanism" outright.
- The non-additive compensation seen with MLP ablations is largely overlap between the MLP and head pathways in the same layers, i.e. within-block redundancy, consistent in kind with the dual-pathway paper's CPA finding.
## Caveats
- Mean-ablation at the last position only; one seed; POPE object questions only; LLaVA's LM was jointly fine-tuned, JEPA's is frozen 4-bit.
- LLaVA AUROC surviving second-half ablation implies the ranking is carried by earlier last-position computation (consistent with the all-ablated sanity check = 0.5); this is an inference, not separately tested.
