# Control A: CLIP projector-only arm (LLaVA-1.5 stage-1 projector + frozen 4-bit Vicuna), 1000 POPE questions, 265 images, 300 image-cluster bootstrap draws

Arms: LLaVA-1.5 (full recipe) | CLIP, projector-only | JEPA, projector-only. Recipe = LLaVA - CLIP. Encoder = CLIP - JEPA.

## T1 behaviour and information
| metric | LLaVA | CLIP | JEPA |
|---|---|---|---|
| output AUROC | 0.937 [0.922,0.952] | 0.907 [0.890,0.923] | 0.818 [0.791,0.845] |
| probe ceiling | 0.969 | 0.943 | 0.887 |
| d' | 2.42 | 1.90 | 1.31 |
| acc | 0.865 | 0.670 | 0.723 |
| hit rate | 0.888 | 0.976 | 0.630 |
| yes-rate | 0.523 | 0.806 | 0.407 |
| false-yes @ hit .89 | 0.166 [0.102,0.270] | 0.258 [0.189,0.321] | 0.494 [0.417,0.558] |
| truthful margin | 2.617 | 0.399 | 0.333 |
| info not used | 0.032 | 0.036 | 0.068 |

## T2 prior-only (noise-corrupted image). CLIP and JEPA share k=0.5; LLaVA k=1.0 calibrated separately (not directly comparable)
| | LLaVA | CLIP | JEPA |
|---|---|---|---|
| corrupted yes-rate | 0.034 | 0.993 | 0.634 |
| R2 clean~corrupted | 0.071 | 0.262 | 0.025 |
| false-yes low/mid/high prior tercile | .114/.159/.200 | .515/.610/.781 | .139/.180/.234 |

## T3 decomposition of LLaVA - JEPA gap
- output AUROC: total +0.119 | recipe +0.030 | encoder +0.089 | recipe share 0.25 [0.13,0.38] -> MOSTLY ENCODER (or its alignment data)
- probe ceiling: total +0.082 | recipe +0.026 | encoder +0.057 | recipe share 0.31 [0.19,0.42]
- d': recipe share 0.47 [0.34,0.59]
- false-yes @ hit .89: total -0.328 | recipe -0.092 | encoder -0.236 | recipe share 0.28 [-0.14,0.48]
- truthful margin: recipe share 0.97 [0.96,0.99] (raw-logit scale; LLaVA's LM was fine-tuned, projector-only arms have compressed logits, so this is a scale effect, not an ordering effect)
- information share of output-AUROC drop: total 0.69 (matches first run), recipe step 0.86, encoder step 0.64

## Reading
1. Rank-based, threshold-free measures (AUROC, probe ceiling, matched-sensitivity false-yes): about 3/4 of the LLaVA-JEPA gap is the CLIP->JEPA step (encoder plus its alignment data), about 1/4 is the joint-fine-tuning recipe.
2. Raw logit margin and default-threshold accuracy are dominated by recipe/calibration: CLIP projector-only is heavily yes-biased (yes-rate 0.81, hit 0.98, false-yes@0 0.64, corrupted yes-rate 0.99).
3. Q4 (does caption alignment make embeddings more prone to language-prior domination?): CLIP (caption-aligned) is MORE yes-biased and more prior-driven (R2 0.26 vs 0.025, higher false-yes in every prior tercile) than JEPA. But at matched sensitivity JEPA hallucinates more (0.494 vs 0.258). So JEPA's hallucination is not prior domination; it is a weaker evidence signal. Caption alignment shifts the operating point toward yes, not the evidence quality.
## Caveats
- CLIP stage-1 projector trained on ~558K captions vs JEPA projector on 5,000: "encoder" share bundles encoder and alignment-data scale (control B, own CLIP projector on 5K captions, not run).
- One projector per arm, one seed; bootstrap covers question sampling only.
- Layout: CLIP [BOS][image][text] (as its projector was trained), JEPA [image][BOS][text].
- LLaVA in fp16, CLIP/JEPA 4-bit; LLaVA corruption strength calibrated separately.
