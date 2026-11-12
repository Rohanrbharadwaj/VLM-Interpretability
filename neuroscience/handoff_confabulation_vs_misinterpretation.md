# Handoff: material from the prior-vs-evidence (LLaVA / IBL) chat that may bear on "confabulation vs misinterpretation"

**Honest note:** that chat never discussed "confabulation vs misinterpretation" by name. Nothing below is a settled idea about it.
Section 1 is established results. Section 2 is my *mapping* of those results onto the distinction, which is a hypothesis to check, not a finding.

## 1. Established results (LLaVA-1.5-7B, 4-bit; Gabor left/right + trial-history prompt)
- Readout: Delta = logit(Right) - logit(Left) at first answer token. Model: Delta = b + beta*s + gamma*pi (s = signed contrast, pi = log-odds of block prior).
- Prompt prior is additive and linear in the count: gamma = 0.277 [0.233, 0.322] (block bootstrap [0.19, 0.36]); side-matched prior effect +0.75 logits [0.64, 0.86], flat across contrast.
- The history sentence costs evidence: beta 15.0 (no history) -> 6.3 (history); grating swing drops to ~0.4-0.5 at every count, even balanced. A neutral sentence of equal length does NOT hurt; the count content does. Placing it before the image is worst (~0.2).
- Boundary shift for 0.2 vs 0.8: LLaVA 12.2% [8.3, 17.5] (history beta) or 5.1% (no-history beta); mouse (IBL, 31 mice) 9.6% [5.4, 15.0].
- Count is linearly decodable from the last-position residual at layers 8-16 (out-of-fold corr 0.32 at L8, 0.75 at L12, 0.95 from L16). Representation is ~linear in count while behavior is a step (flat for 1-7, jump at 8-10).
- Conflict trials (prior points against the stimulus): VLM margin gap ~0.5-0.85 logits at every contrast (constant offset); mouse accuracy conflict vs congruent: 0.78 vs 0.53 at 6% contrast, 0.93 vs 0.87 at 100%.
- Retracted/not supported: A2 "projection explains per-trial shift beyond the count" (artifact); "suppression begins where the count arrives"; "format cost = any extra text".
- Pending (notebook B, running): whether the count direction is causally used (patching hist->none, steering with random/shuffled controls), and which wording keeps the prior with less evidence loss.
- POPE reference: encoder-swap margin gaps LLaVA 5.23, JEPA 0.67.

## 2. Possible mapping (hypothesis only)
- "Prior-driven" error = answer follows pi (the count) against the evidence; "evidence-misread" error = the stimulus signal itself is degraded (beta falls) so the answer is noisier.
- Our data separates the two parameters: gamma (prior pull) vs beta (evidence gain). The history prompt raises gamma AND lowers beta, so both mechanisms are present together; they can be separated per trial.
- A confabulation-like test: trials where evidence is weak (contrast 0-0.045) and the answer follows the prior vs trials where evidence is strong but beta-loss flips the answer.
- Whether this maps onto POPE-style hallucination is untested; it needs a causal direction from B first.

## 3. Open questions to carry over
- Is the prior effect a shift of the decision boundary (confabulation-like) or a loss of evidence gain (misinterpretation-like), or both? Current answer: both, in the history prompt.
- Do other wordings dissociate them (B0 screen: sentence / short / letters / sequence at 2, 5, 8 of 10)?
- Does the same split hold in a second model and on POPE-style data?
