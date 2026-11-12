# Brain hallucinations and VLM hallucinations: chat summary

17 October 2026. Context: UGP-1 (localising attribute and relation hallucination in LLaVA, Qwen2.5-VL and InternVL3).

---

## 1. Hallucination in the brain and its link to VLMs

### Leading account: predictive coding

- Perception combines prior beliefs with sensory evidence. Hallucinations arise when the balance tips.
- Strong-prior hypothesis (Corlett et al. 2019): overly strong, overly precise priors can produce hallucinations even in healthy people. People who hallucinate in daily life are more prone to lab-induced versions.
- Conditioned hallucinations (Powers, Mathys and Corlett 2017): people learn to pair a light with a tone, then later "hear" the tone when only the light appears. A fitted parameter weighting prior against evidence predicts who does this.
- Contested: some studies find weaker priors in psychosis, others stronger. The proposed resolution is that priors differ by modality and by level of the hierarchy.
- Charles Bonnet syndrome: vivid hallucinations in people losing their sight. Reichert, Series and Storkey (NeurIPS 2010) modelled it as a generative network with homeostasis, which hallucinates when input is missing. This is the closest analogue to a VLM given weak visual evidence.

### Confabulation is a separate phenomenon

- "Forgetting context" in dementia, Korsakoff syndrome, brain injury and stroke is confabulation: producing false content without knowing it is false.
- Confabulation is usually explained as a failure of monitoring and verification during retrieval, not a perceptual prior.
- Several papers argue that LLM "hallucination" is really confabulation. For a VLM answering yes to an absent object, this is mostly right: nothing is perceived, a language prior outweighs weak visual evidence, and nothing checks the answer.

### Link to the UGP results

- DLA finding: a false yes uses the same Yes/No gate as a true yes, with a smaller margin, and the cause is upstream. This fits the strong-prior account: no dedicated hallucination circuit, just weak evidence meeting a normal prior at a normal decision stage.
- Testable prediction: hallucination rate should track the strength of the language prior relative to the visual signal. The blind-model lean in the noise-corruption runs is effectively a measured prior.
- A safe writeup claim is "consistent with a strong-prior account", not "the same mechanism as the brain".
- The comparison only holds at the computational level, not because "both are networks of neurons".

### Bibliography

- Corlett, Horga, Fletcher, Alderson-Day, Schmack, Powers. Hallucinations and Strong Priors. Trends in Cognitive Sciences 23(2), 2019. https://repository.cshl.edu/id/eprint/37532/
- Powers, Mathys, Corlett. Pavlovian conditioning-induced hallucinations result from overweighting of perceptual priors. Science, 2017. https://pmc.ncbi.nlm.nih.gov/articles/PMC5802347
- Sterzer et al. The Predictive Coding Account of Psychosis. Biological Psychiatry 84(9), 2018. https://discovery.ucl.ac.uk/id/eprint/10057511/
- Reichert, Series, Storkey. Hallucinations in Charles Bonnet Syndrome Induced by Homeostasis: a Deep Boltzmann Machine Model. NeurIPS 2010. https://neurips.cc/virtual/2010/poster/2210
- Smith et al. Hallucination or Confabulation? Neuroanatomy as metaphor in Large Language Models. PLOS Digital Health, 2023. https://pmc.ncbi.nlm.nih.gov/articles/PMC10619792/
- Sui, Duede, Wu, So. Confabulation: The Surprising Value of Large Language Model Hallucinations. ACL 2024. https://preview.aclanthology.org/setup/2024.acl-long.770
- Barros. I Think, Therefore I Hallucinate. arXiv:2503.05806, 2025. https://arxiv.org/pdf/2503.05806
- Ardoin, Cai, Wunder. Where Confabulation Lives: Latent Feature Discovery in LLMs. EMNLP 2025. https://aclanthology.org/people/gerhard-wunder/unverified/

---

## 2. Research connecting VLMs and the brain

Almost all of it compares representations, not circuits. No paper matches a causal VLM circuit to a brain circuit. That gap is open.

### Bavaresco et al., EACL 2026: Vision-Language Models Align with Human Neural Representations in Concept Processing

- **Data.** Pereira 2018 fMRI: 16 people, 180 concept words, each shown in sentences or next to pictures.
- **Brain regions.** Left-hemisphere language network and the visual network.
- **Models.** Six VLMs in three families:
  - contrastive: CLIP, ALIGN
  - encoders: VisualBERT, LXMERT
  - generative: IDEFICS2, LLaVA-NeXT
- **Text-only comparisons.** BERT, Mistral, Llama3, GloVe.
- **Method.** RSA: build concept-by-concept distance matrices for brain and each model layer, then correlate them with Spearman's rho. Best layer reported.
- **Results.**
  - VLMs beat text-only models, most clearly in the picture condition (VisualBERT reaches 0.48 in the visual network).
  - Vision-language encoders align better than generative VLMs.
- **Ablations.**
  - Without the picture, CLIP, ALIGN, VisualBERT and LLaVA drop sharply. Only LXMERT and IDEFICS2 hold up.
  - In the visual network, part of the generative VLMs' alignment was already present in their text backbones before vision training.
- **Caveats.** Models differ in size, data and objective at once, so no single factor can be blamed. Brain matrices are averaged across participants.

### Wu et al., arXiv:2605.28818, May 2026: Do VLMs Align Better with Humans than LLMs during Natural Reading?

- **Design.** Matched LLM/VLM pairs on text-only input, so no image is seen at test time:
  - Qwen2 vs Qwen2-VL
  - DeepSeek-LLM vs DeepSeek-VL
  - Mistral vs Idefics2 (more loosely matched)
- **Human data.** Whole-cortex fMRI plus eye tracking of regressive saccades (jumps back to re-read).
- **Results.**
  - Whole-cortex fMRI shows no broad VLM advantage.
  - VLM attention predicts re-reading better in the two clean pairs (p = .0072 and .0020), not in Idefics2.
  - The VLM's fMRI advantage grows with how visually evocative the sentence is.
- **Conclusion.** Multimodal training helps selectively, not globally. The authors do not claim it as the cause.

### Sun, Lu et al., arXiv:2607.00397: NeuroCogMap Reveals Cognitive Organization of Large Language Models

- **Models.** Gemma2-2B, Gemma2-9B-IT and Llama-3.1-8B. Text-only; VLMs are listed as future work.
- **Method.**
  1. Collect 45 capabilities from the literature, each tied to benchmark datasets.
  2. Cluster residual-stream SAE features by their task response profiles into 270 parcels.
  3. Label each parcel with an LLM, build a directed connectome between parcels, and link parcels to capabilities.
  4. Sort capabilities into four levels: perception, representation, abstraction, application.
- **Hallucination has two forms.**
  - TruthfulQA: retrieval without evaluative control.
  - NQ-Open: retrieval modules that fail to coordinate.
- **Refusal failure and sycophancy.** Classed as behavioural-control failures, not representational ones.
- **Detection.** Hallucination AUROC 0.681 (Gemma2-2B) and 0.840 (Gemma2-9B-IT). Refusal failure about 0.99.
- **Steering.** Raises AdvBench refusal from 38.3 percent to 98.6 percent on Gemma2-2B.
- **Brain.** Parcels predict story-listening fMRI at r = 0.407, against 0.367 for raw hidden states, with the biggest gains in default and control networks.
- **Caveats.** Labels and matching are judged mostly by an LLM (GPT-OSS-20B). Many free hyperparameters. The detector runs after generation.

### Konrad, Tanyel, Ayvaz, ICML 2026 MI Workshop: A Path Already Walked

- Position paper with no transformer experiments.
- Argues for importing network-neuroscience tools (graph measures, null models, known failure modes) into circuit analysis in a disciplined way, not as loose analogy.
- Gives a "graph contract", a mapping from neuroscience concepts to transformer analyses, an effective-connectivity proxy for gated MLPs, and eight testable translations with failure criteria.
- Only the abstract was readable here. The full paper is worth reading before path patching.

### Other alignment work

- Oota et al., ICLR 2025: instruction-tuned MLLMs fit visual cortex better than a vision-only ViT. Middle layers align with high-level visual areas.
- Oota et al., 2025, video and audio MLLMs: the mapping is hierarchical, with early layers matching early sensory cortex.
- arXiv:2603.08303, EEG study of 32 LVLMs: layers 8 to 16 peak in alignment with brain activity 100 to 300 ms after the image. Architecture matters 3.4 times more than scale.
- Formal vs functional circuits, arXiv:2503.11302: little circuit overlap between formal and functional language tasks in LLMs, but also little between formal tasks, so the brain's dissociation does not cleanly reappear.

### Relevance to the UGP

- The finding that LLaVA's alignment is fragile without images fits the "weak visual signal, strong language prior" picture.
- Path patching builds the kind of graph "A Path Already Walked" discusses, and the permutation-based material rule plays the role of its null models.
- NeuroCogMap leaves VLMs open, and UGP components are established causally, not just by correlation.

### Bibliography

- Bavaresco, de Heer Kloots, Pezzelle, Fernández. Vision-Language Models Align with Human Neural Representations in Concept Processing. EACL 2026. https://arxiv.org/html/2407.17914v3
- Wu, Ma, Li, Tang, Lu. Do VLMs Align Better with Humans than LLMs during Natural Reading? arXiv:2605.28818, 2026. https://papers.cool/arxiv/2605.28818
- Sun, Lu et al. NeuroCogMap Reveals Cognitive Organization of Large Language Models. arXiv:2607.00397, 2026. https://arxiv.org/pdf/2607.00397
- Konrad, Tanyel, Ayvaz. A Path Already Walked: On Inheriting Network-Neuroscience Tools for Mechanistic Interpretability. ICML 2026 MI Workshop. https://icml.cc/virtual/2026/79393
- Oota et al. Correlating instruction-tuning (in multimodal models) with vision-language processing (in the brain). ICLR 2025. https://arxiv.org/pdf/2505.20029v1
- Oota et al. Multimodal video and audio MLLM brain encoding. arXiv:2506.08277, 2025. https://www.arxiv.org/pdf/2506.08277
- LVLM–EEG alignment across 32 models. arXiv:2603.08303, 2026. https://arxiv.org/html/2603.08303v1
- Are formal and functional linguistic mechanisms dissociated in language models? arXiv:2503.11302, 2025. https://arxiv.org/pdf/2503.11302

---

## 3. How papers connect AI and the brain, and how the brain is analysed

### Four bridges, weakest to strongest

1. **Same stimuli, compare activity.** Encoding models, RSA or decoding. Shows the two systems hold similar information, not that they use the same computation. Most papers stop here.
2. **Same task, compare behaviour.** Do the model's errors, reaction times or eye movements match humans'? Matching errors reveal more about the algorithm than matching successes.
3. **Shared computational theory.** Fit the same model, such as prior × likelihood, to both systems and compare the parameters. This is Marr's computational level and the most defensible bridge for hallucination.
4. **Network as model organism.** Use the network to generate hypotheses, then test them on people.

### Brain methods and their UGP analogues

| Brain method | Gives | UGP analogue | Limitation |
|---|---|---|---|
| Lesions | Necessity of a region | Ablation | Uncontrolled, large, with compensation |
| TMS, cooling, drugs | Temporary causal disruption | Coarse activation patching | Centimetre scale |
| Optogenetics (animals) | Cell-type control with millisecond timing | Activation patching | Mostly rodents |
| fMRI | Location, 1 to 2 s blur | Per-layer maps | Too slow for information flow |
| EEG and MEG | Millisecond timing | Layer depth as time | Poor localisation |
| Intracranial recordings | Precise timing and location in humans | Residual stream at a few positions | Sparse, clinical only |
| Neuropixels (animals) | Thousands of neurons | Neuron or SAE features | Tiny fraction of the brain |
| Decoding over time | When information becomes decodable | Per-layer probes | Decodable does not mean used |
| DCM, Granger causality | Directed influence between regions | Path patching | Inferred from a model, not intervention |
| Anatomical tracing | Wiring | Weights | Wiring is not function |

The key asymmetry: the brain cannot be rerun in the same state with one component swapped, so the clean counterfactual behind activation patching is unavailable. Recurrence also makes "where information first appears" much harder to define than in a single forward pass.

### Where false information first appears

- **Timing contrasts on the same input** (the same design as false_yes vs true_no):
  - False alarms on noisy stimuli resemble real detections in sensory cortex, often with bias already present before the stimulus.
  - In conditioned hallucinations, auditory cortex activates on false reports, scaled by the weight on the prior.
  - Auditory verbal hallucinations involve speech areas, consistent with inner speech mistaken for an outside voice. The normal suppression of the auditory response to one's own voice is weaker in schizophrenia (Ford and Mathalon).
- **Failed checking stage:**
  - Schnider's "reality filter": an early orbitofrontal signal, about 200 to 300 ms after a cue, marks whether a memory belongs to the present. Confabulation follows when it fails.
  - DRM false memories use the normal retrieval route but carry less sensory reinstatement than true memories.
- **Lesion mapping:**
  - Sensory loss produces false percepts.
  - Frontal monitoring damage produces confabulation.

Common pattern: false content is produced by normal machinery, driven by expectation and not caught by monitoring. There is no dedicated error circuit. This matches the DLA reading.

### Idea for the UGP

Test for a missing check: a component that, in true_no cases, pushes against a prior-driven Yes and is weaker in false_yes cases. Candidates to probe with exact patching are LLaVA's L28 MLP suppressor and Intern's L27 MLP subtraction.

### Bibliography

- Schnider. Spontaneous confabulation, reality monitoring and the limbic system. Brain/Nature Reviews Neuroscience work, 2000s (orbitofrontal reality filtering).
- Ford, Mathalon. Corollary discharge dysfunction in schizophrenia (N1 suppression studies), 2000s.
- Roediger, McDermott. Creating false memories (DRM paradigm). Journal of Experimental Psychology, 1995.
- Kriegeskorte, Mur, Bandettini. Representational similarity analysis. Frontiers in Systems Neuroscience, 2008.
- Marr. Vision. 1982 (levels of analysis).
- Powers, Mathys, Corlett, 2017 (see Section 1).

Note: the Section 3 neuroscience references come from general knowledge, not this chat's searches. Check exact titles and years before citing.
