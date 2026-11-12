## Step 4: last-position ablations (noise control run, 1000 balanced POPE questions, 69 configs)

**Setup.** Each config zeroes (×0.0) or halves (×0.5) the last-position output of one component or a group, and the Yes − No logit difference $\Delta$ is recomputed. Baseline: accuracy 0.723, yes-rate 0.407, false-yes rate 0.184 (halluc), miss rate 0.370, mean $\Delta = -0.126$. AUROC is the threshold-free measure: a pure bias shift changes accuracy, false-yes and miss, but not AUROC.

**Headline.** Ablating one component at the last position moves the answer threshold but barely changes how well the model separates present from absent objects.

### 1. Single ablations change calibration, not discrimination

AUROC stays at 0.809–0.821 for every single-component ablation, while the yes-rate swings between 0.09 and 1.0.

| Config | Accuracy | Yes-rate | Mean $\Delta$ | AUROC |
|---|---|---|---|---|
| L28.mlp ×0 | 0.500 | 1.000 | +1.911 | 0.809 |
| L31.mlp ×0 | 0.584 | 0.088 | −1.606 | 0.815 |
| L20.mlp ×0 | 0.666 | 0.206 | −0.520 | 0.816 |
| L18.mlp ×0 | 0.709 | 0.637 | +0.198 | 0.811 |
| L26H3 ×0 | 0.719 | 0.361 | −0.183 | 0.821 |

Zeroing L28.mlp makes the model answer "yes" to everything, and AUROC is still 0.809. These components set where the threshold sits. The evidence that separates present from absent is not in any single last-position component.

### 2. DLA direct effect predicts the ablation effect poorly

The correlation between the DLA prediction $-\overline{\text{contribution}}$ and the measured change in $\Delta$ is 0.721, but it is carried by two points (L28.mlp and L20.mlp) and $n=10$. Predicted values are read off the scatter plot, so they are approximate. Measured values are the table's mean $\Delta$ minus the baseline of $-0.126$.

| Component | DLA predicts | Measured | Note |
|---|---|---|---|
| L26H3 | −1.55 | −0.06 | almost fully compensated downstream |
| L20.mlp | −3.1 | −0.39 | about 87% compensated |
| L28.mlp | +3.7 | +2.04 | about 45% compensated |
| L18.mlp | +1.4 | +0.32 | |
| L27.mlp | +1.0 | −0.16 | sign flips |
| L31.mlp | −0.8 | −1.48 | larger than predicted |

- L31.mlp is the last layer, so there is no downstream compensation. The extra effect is probably the final layer-norm denominator changing when its output is zeroed. This is a guess and has not been checked.
- The L27.mlp sign flip fits a chain in which its output feeds L28.mlp.

### 3. Group ablations behave the same way

| Group | Accuracy | False-yes | Miss | AUROC |
|---|---|---|---|---|
| baseline | 0.723 | 0.184 | 0.370 | about 0.82 (see note) |
| hallucination-pathway top 3, ×0 | 0.543 | 0.002 | 0.912 | 0.819 |
| hallucination-pathway top 5, ×0 | 0.518 | 0.002 | 0.962 | 0.815 |
| hallucination-pathway top 10, ×0 | 0.500 | 0.000 | 1.000 | 0.796 |
| grounding-pathway top 3, ×0 | 0.500 | 1.000 | 0.000 | 0.811 |
| grounding-pathway top 5, ×0 | 0.500 | 1.000 | 0.000 | 0.765 |
| grounding-pathway top 10, ×0 | 0.500 | 1.000 | 0.000 | 0.722 |

- The hallucination-pathway top 3 cuts false-yes from 0.184 to 0.002, but misses rise from 0.37 to 0.91 and accuracy drops from 0.72 to 0.54. It is a bias shift, not a selective fix. AUROC does not move.
- The grounding groups push everything to "yes", so accuracy, false-yes and miss are saturated and uninformative. Only AUROC tells you anything there.
- Baseline AUROC is not printed in the table (the baseline row is shown only by its accuracy and rates). The single-ablation values suggest about 0.82.

### 4. Where AUROC does drop

Only the grounding top 5 at ×0 (0.765) and top 10 at ×0 (0.722), where several components are removed at once. At ×0.5 it stays at about 0.81. So the image evidence is spread across many components.

The random matched sets have large variance, because some draws hit L28.mlp (for example, $k=5$, ×0: $\Delta$false-yes $=+0.465 \pm 0.333$). Their AUROC is not printed. Check `ablation_table.csv` before calling the 0.72 meaningful.

### Caveats

- Zero-ablation removes each component's constant offset too, which is most of what moves here. Step 6 (mean-ablation, with AUROC and bootstrap CIs) separates the offset from the evidence and tests whether the evidence MLPs are needed.
- AUROC at $n=1000$ has a standard error of about 0.015, so differences within 0.809–0.821 are within noise.
- Last-position ablation only removes a component's output at the final token. It does not touch how that component's earlier-position outputs reach the last position through attention, which is what step 7's K/V patching addresses.

---

## CORRECTION (added after the random-control AUROCs were available)

Section 4 above ("Where AUROC does drop") is **retracted as a finding about the pathways**. The matched random sets in `ablation_table.csv` drop AUROC just as much:

| Config (×0) | Pathway AUROC | Random sets (3 draws) |
|---|---|---|
| grounding top 5 | 0.765 | 0.763, 0.802, 0.755 |
| grounding top 10 | 0.722 | 0.607, 0.798, 0.748 |
| hallucination top 10 | 0.796 | 0.810, 0.806, 0.797 |

Baseline AUROC is 0.818. Zero-ablating 5–10 arbitrary components can lower AUROC to 0.61–0.76, so the grounding drops are not pathway-specific, and the claim that "the image evidence is spread across many components" does not follow from step 4. What *is* pathway-specific in step 4 is the bias shift (for example hallucination top 3: false-yes 0.002 vs random 0.37 / 0.186 / 0.184), and even that depends on whether a random set happens to include L28.mlp or L20.mlp. Discrimination claims should rest on step 6 (mean-ablation with 10 matched random sets).
