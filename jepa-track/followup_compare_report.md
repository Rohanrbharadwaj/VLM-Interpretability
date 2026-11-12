# Encoder swap: LLaVA-1.5 vs JEPA-Vicuna (auto-generated)

## Verdict

- Mechanism (1/4 criteria): DIFFERENT mechanism. Parts of the computation differ between the two models; see the failed criteria.
- Information reaching the language model: JEPA's residual stream carries significantly LESS decodable object information than LLaVA's, so the JEPA encoder/projector is at least partly responsible for the weaker discrimination (encoder-driven component).
- Use of the information: at least one model's output already uses nearly all the information its residual stream contains.
- Caveats: LLaVA's language model was fine-tuned and ours is frozen; the two noise corruptions are calibrated separately (to chance AUROC); the probe is supervised, so 'decodable' means linearly decodable given labels; prompts differ; POPE object questions only (the task's attribute/relation questions are not covered unless HALLUC_TYPE was changed).

## Criteria

| criterion                                                                                                               | passed   | detail                                                                                      |
|:------------------------------------------------------------------------------------------------------------------------|:---------|:--------------------------------------------------------------------------------------------|
| M1  most necessary MLP window agrees (top window equal, or Spearman over 8 windows >= 0.6)                              | False    | top window LLaVA window MLPs 28-31 vs JEPA window MLPs 16-19; Spearman -0.38                |
| M2  the JEPA-derived evidence MLPs drop LLaVA's AUROC more than random sets (z <= -2 or below every random set)         | False    | LLaVA d_auroc -0.002, z -0.8 (JEPA -0.057, z -5.9)                                          |
| M3  K/V route                                                                                                           |          | not available (patching cell not run)                                                       |
| M4  direct-effect signature similar (noise-corrected Spearman >= 0.6) with the same-sign L20/L28 constants              | False    | rho +0.29 (0.25, 0.32), corrected 0.33; L20 const +2.53/+3.11, L28 -2.58/-3.70              |
| M5  evidence and opposing MLP sums are anti-correlated within class (r <= -0.3) in both models (compensation signature) | True     | LLaVA: -0.97/-0.98 | JEPA: -0.92/-0.92                                                      |
| I1  JEPA's probe ceiling is LOWER than LLaVA's (CI excludes 0): less decodable object information reaches the LM        | True     | JEPA - LLaVA ceiling -0.082 [-0.100, -0.064] (LLaVA 0.969, JEPA 0.887)                      |
| I2  both models leave decodable information unused (ceiling - output AUROC: CI lower bound > 0.02)                      | False    | gap LLaVA +0.032 (0.019, 0.044), JEPA +0.068 (0.046, 0.088)                                 |
| I3  JEPA's probe recovers a larger share of its own errors than LLaVA's does (LM-side share of errors)                  | False    | share of model errors the probe gets right: JEPA 0.59 (0.53, 0.65), LLaVA 0.59 (0.51, 0.68) |
| I4  LLaVA separates the classes better than JEPA (AUROC difference CI excludes 0)                                       | True     | AUROC LLaVA - JEPA +0.119 (0.095, 0.145); d' 2.42 vs 1.31                                   |

## Behaviour (same questions)

      model    subset    n  auroc  dprime   acc  yes_rate   gap    sd  halluc  miss  corr_auroc
  LLaVA-1.5  all 1000 1000  0.937   2.422 0.865     0.523 5.234 3.392   0.158 0.112       0.624
  LLaVA-1.5 first 400  400  0.927   2.310 0.842     0.565 5.146 3.410   0.209 0.110       0.589
  LLaVA-1.5 first 300  300  0.929   2.321 0.847     0.567 5.178 3.423   0.204 0.108       0.587
JEPA-Vicuna  all 1000 1000  0.818   1.310 0.723     0.407 0.666 0.607   0.184 0.370       0.553
JEPA-Vicuna first 400  400  0.812   1.282 0.700     0.422 0.661 0.612   0.209 0.383       0.519
JEPA-Vicuna first 300  300  0.818   1.317 0.703     0.437 0.688 0.624   0.218 0.367       0.516

## Mean-ablation (change in AUROC; random controls)

                      config model  d_auroc  ci_lo  ci_hi  rand_mean  rand_sd  rand_min      z  below_min  d_mean_logit
               evidence MLPs LLaVA   -0.002 -0.005  0.003      0.000    0.003    -0.004 -0.796      False        -0.333
               evidence MLPs  JEPA   -0.057 -0.074 -0.037     -0.006    0.009    -0.021 -5.877       True        -0.012
               opposing MLPs LLaVA   -0.000 -0.001  0.001     -0.000    0.003    -0.004  0.043      False         0.016
               opposing MLPs  JEPA    0.005  0.001  0.010     -0.009    0.008    -0.025  1.719      False        -0.001
        answer-reactive MLPs LLaVA   -0.003 -0.005 -0.001      0.000    0.001    -0.001 -3.400       True         0.037
        answer-reactive MLPs  JEPA   -0.018 -0.030 -0.008     -0.002    0.006    -0.015 -2.424       True        -0.003
              evidence heads LLaVA    0.000 -0.002  0.003     -0.000    0.000    -0.001  1.197      False        -0.102
              evidence heads  JEPA   -0.011 -0.022  0.000     -0.000    0.001    -0.003 -7.206       True         0.008
                  bias heads LLaVA   -0.000 -0.001  0.000     -0.000    0.000    -0.000  0.049      False         0.018
                  bias heads  JEPA    0.002 -0.000  0.004     -0.000    0.001    -0.002  1.334      False         0.003
       all MLPs, second half LLaVA   -0.000 -0.006  0.006      0.002    0.006    -0.008 -0.420      False        -0.067
       all MLPs, second half  JEPA   -0.202 -0.247 -0.157     -0.078    0.042    -0.142 -2.929       True        -0.016
2nd half minus evidence MLPs LLaVA    0.004 -0.000  0.009      0.001    0.004    -0.005  0.829      False         0.248
2nd half minus evidence MLPs  JEPA   -0.016 -0.033 -0.002     -0.013    0.010    -0.026 -0.362      False        -0.046
             window MLPs 0-3 LLaVA   -0.000 -0.001  0.000     -0.001    0.003    -0.004  0.074      False        -0.000
             window MLPs 0-3  JEPA    0.002  0.000  0.004     -0.015    0.013    -0.038  1.306      False        -0.002
             window MLPs 4-7 LLaVA   -0.001 -0.001  0.000     -0.001    0.003    -0.004 -0.021      False         0.000
             window MLPs 4-7  JEPA    0.005  0.001  0.009     -0.015    0.013    -0.038  1.523      False        -0.002
            window MLPs 8-11 LLaVA   -0.000 -0.002  0.001     -0.001    0.003    -0.004  0.138      False         0.028
            window MLPs 8-11  JEPA    0.002 -0.005  0.007     -0.015    0.013    -0.038  1.252      False         0.021
           window MLPs 12-15 LLaVA    0.009  0.004  0.016     -0.001    0.003    -0.004  3.553      False         0.081
           window MLPs 12-15  JEPA   -0.032 -0.042 -0.022     -0.015    0.013    -0.038 -1.226      False        -0.020
           window MLPs 16-19 LLaVA    0.001 -0.002  0.005     -0.001    0.003    -0.004  0.717      False        -0.317
           window MLPs 16-19  JEPA   -0.054 -0.071 -0.037     -0.015    0.013    -0.038 -2.940       True        -0.012
           window MLPs 20-23 LLaVA    0.001 -0.001  0.003     -0.001    0.003    -0.004  0.518      False         0.081
           window MLPs 20-23  JEPA   -0.008 -0.013 -0.002     -0.015    0.013    -0.038  0.556      False         0.002
           window MLPs 24-27 LLaVA    0.000 -0.001  0.001     -0.001    0.003    -0.004  0.319      False         0.022
           window MLPs 24-27  JEPA    0.004  0.001  0.009     -0.015    0.013    -0.038  1.470      False        -0.012
           window MLPs 28-31 LLaVA   -0.004 -0.007 -0.001     -0.001    0.003    -0.004 -1.149      False         0.153
           window MLPs 28-31  JEPA   -0.040 -0.058 -0.021     -0.015    0.013    -0.038 -1.817       True        -0.041
             window MLPs 0-7 LLaVA   -0.001 -0.002  0.000      0.003    0.005    -0.005 -0.839      False        -0.002
             window MLPs 0-7  JEPA    0.006  0.002  0.010     -0.027    0.017    -0.052  1.937      False        -0.002
            window MLPs 8-15 LLaVA    0.008  0.003  0.015      0.003    0.005    -0.005  0.960      False         0.081
            window MLPs 8-15  JEPA   -0.040 -0.052 -0.027     -0.027    0.017    -0.052 -0.752      False        -0.024
           window MLPs 16-23 LLaVA   -0.000 -0.004  0.004      0.003    0.005    -0.005 -0.796      False        -0.256
           window MLPs 16-23  JEPA   -0.071 -0.090 -0.051     -0.027    0.017    -0.052 -2.543       True        -0.004
           window MLPs 24-31 LLaVA   -0.001 -0.004  0.003      0.003    0.005    -0.005 -0.821      False         0.158
           window MLPs 24-31  JEPA   -0.024 -0.040 -0.008     -0.027    0.017    -0.052  0.191      False        -0.041