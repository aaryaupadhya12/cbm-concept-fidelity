# derm-concept-audit

**Do vision-language concept scores agree with dermatologists? An expert-label audit of label-free concept bottleneck models.**

Working title for the paper:

> *Expert-Grounded Audit of Vision-Language Concept Labels for Dermatological Concept Bottleneck Models*

Alternates under consideration:
- *When the Ground Truth Is the Model: Auditing VLM Concept Labels Against Dermatologist Annotations*
- *Concept Source Dominates Head Architecture in Dermatological CBMs*

---

## What this is

Label-free CBMs (LF-CBM, VLG-CBM, LaBo, f-CBM) all generate concept labels from VLM text-image
similarity, and all report faithfulness metrics computed against those same VLM-derived labels.
That is circular, and nobody has tested the underlying assumption in a domain where real expert
concept annotations exist.

Dermatology has them. This repo audits the assumption.

**Scope.** Derm7pt is the primary dataset (7-point checklist, dermoscopic, expert-graded).
PH2 is the V2 extension (5 dermoscopic features + ABCD, with lesion masks). Everything in V1
runs on Derm7pt only.

**Everything is frozen.** No fine-tuning anywhere. Frozen VLM backbone, cached features,
probes on top.

---

## Hypotheses

Locked before experimentation. Do not edit after Step 2 begins.

| ID | Statement | Status |
|----|-----------|--------|
| **H1** | VLM concept scores agree with dermatologist annotations at levels that vary substantially by concept. Prompt construction accounts for a non-trivial share of that variation. | Primary |
| **H2′** | At a 7-concept medical bottleneck, the KAN head's reported leakage advantage does not transfer. Head architecture changes leakage less than concept source does. | Merged from old H2 |
| **H3** | Concept source dominates head architecture. Swapping VLM concept labels for expert labels moves accuracy more than any head swap. | Headline |
| **H4** | A CBM trained on VLM concept labels shows a larger gap between NEC-vs-pseudo-labels and NEC-vs-expert-labels than one trained on expert labels. Standard evaluation flatters it. | Faithfulness |

### Note on H1's direction

An earlier version of H1 predicted that derm-pretrained backbones beat general CLIP. That
direction is contested: the PanDerm authors found CLIP-large outperformed both BiomedCLIP and
MONET as a teacher model, attributing it to the limited scale of skin imagery in medical-domain
CLIP training. H1 is therefore phrased as a test, not a prediction. If the derm-specific backbone
loses, that is a reportable result and PanDerm is the citation.

### Note on H2′ (why the old H2 was retired)

The old H2 predicted Linear/MLP/FastKAN would tie on accuracy. That is close to what f-CBM
already reports — their KAN is presented as a leakage-mitigation device that costs a small amount
of task accuracy, not as an accuracy improvement. Registering their published result as our null
would invite the exact reviewer comment we were trying to avoid.

What is genuinely untested is whether the leakage benefit survives our regime. Their ablation runs
on N24News (multimodal news classification) with a CLIP-base backbone. Ours is dermoscopy, seven
correlated checklist criteria, ~800 training images. Leakage depends on bottleneck width and
concept correlation, so the transfer is an open question. Accuracy becomes a reported control
column, not a claim.

---

## The five steps

### Step 1 — Cache and controls

Encode Derm7pt dermoscopic images once with ViT-L/14 (openai), fp16 on GPU. Save embeddings and
index as a Kaggle Dataset so the encode never repeats. Fix `encode_texts` reloading the model on
every call.

**Controls added here:**
- Concept-irrelevant prompt pair (`"a photo of a bicycle"` / `"a photo of a chair"`) — empirical AUROC floor
- Zero-shot melanoma vs nevus with the same backbone and protocol

**Gate.** Irrelevant pair must sit near 0.50. If it reads 0.65, stop and find the leak or sign error.

**Cost.** 1–2 min GPU. Embeddings are ~3 MB. Everything after this is CPU.

---

### Step 2 — H1: the prompt grid

This is the backbone of the project. Twelve cells:

```
{ref-prompt, single antonym, antonym set} × {short label, clinical definition} × {ensemble on/off}
```

**Scoring.** Softmax, not a raw cosine difference. MONET Eq. 1:

```
p(i,c) = exp(sim(Iᵢ, T_c)/λ) / [exp(sim(Iᵢ, T_c)/λ) + exp(sim(Iᵢ, T_r)/λ)]
```

λ fixed at 1 and recorded in the protocol JSON. AUROC is invariant to λ under a single negative,
so this costs nothing at H1 — but scaling matters once scores feed a head, and λ is a knob the
literature admits to tuning. Fix it now.

**Negative constructions:**

| Name | Negative is | Source |
|------|-------------|--------|
| `ref` | the template with the concept slot removed | MONET Eq. 1 |
| `antonym` | one clinical opposite | our original |
| `antonym_set` | 2–4 opposites, softmax over all | MONET Eq. 2 |

**Templates.** Dermoscopy register only. Derm7pt derm images are dermoscopic; MONET uses separate
template sets per modality, and mixing clinical-register templates ("a photo of") into a dermoscopy
ensemble is a confound we would then have to explain.

**Binarisation.** Run twice — atypical-vs-rest (clinical convention) and atypical-vs-typical
(absent dropped). Second goes in an appendix. If they diverge, that says something about what the
score tracks.

#### Output table

| Concept | ref+short | ref+clin | ant+short | ant+clin | set+short | set+clin | +ens (×6) | **spread** | n_pos |
|---|---|---|---|---|---|---|---|---|---|
| pigment_network | | | | | | | | | |
| blue_whitish_veil | | | | | | | | | |
| vascular_structures | | | | | | | | | |
| pigmentation | | | | | | | | | |
| streaks | | | | | | | | | |
| dots_and_globules | | | | | | | | | |
| regression_structures | | | | | | | | | |
| *mean* | | | | | | | | | |
| CONTROL: irrelevant | ~0.50 | | | | | | | | |
| CONTROL: zero-shot mel | | | | | | | | | |

Cells carry AUROC with `[ci_lo, ci_hi]` from 500 bootstrap resamples.

**The spread column is the first contribution.** Everything else in the table is measurement.
If spread on any concept is large, then "the VLM concept label" is not a well-defined object, and
every faithfulness metric computed against one is measuring the prompt as much as the model.

**Selection rule, written before results are viewed:** all 12 cells evaluated on validation; the
cell with highest mean AUROC across concepts is selected; test reported once for that cell; all
other cells reported on validation only.

**Sanity anchors.** DermFM-Zero reports pigment network at roughly 0.52 for CLIP and 0.80 for
MONET. On SkinCon concepts, MONET reaches mean AUROC 0.766 against CLIP's 0.692, with 19 of 21
concepts above 0.7 versus CLIP's 9. **A CLIP result near chance on pigment network is probably
correct, not broken.** A result near 0.75 from vanilla CLIP should be treated as a suspected bug
before it is treated as a finding.

---

### Step 3 — Baselines

| Method | Input | Bal. acc | AUROC | Role |
|---|---|---|---|---|
| Majority class | — | | 0.500 | Floor |
| Linear probe | 768-d embedding | | | **Parity target** |
| CBM, VLM concepts | 7-d scores | | | Our model |
| CBM, expert concepts | 7-d binaries | | | Concept ceiling |
| CBM, shuffled concepts | 7-d permuted | | ~0.50 | Sanity |

Our claim is *best faithfulness at parity accuracy* — match a frozen-backbone black-box probe
within a point or two, win on the faithfulness axis. Row 2 is what that claim is measured against,
so it is not optional.

**Gate.** Shuffled-concept baseline must collapse to chance. If a shuffled bottleneck still
predicts melanoma, the head is reading something other than the concepts.

**ABC baseline is deferred to V2.** It needs lesion masks, Derm7pt has none, and auto-segmenting
would add a segmentation-quality confound to the only clean comparison in the paper. PH2 has masks;
this is one of the two reasons PH2 is the V2 dataset.

---

### Step 4 — H3: concept source × head

Two 7-dim input matrices (VLM scores from the Step 2 winner; expert binaries from `meta.csv`)
crossed with three heads.

| Concept source | Linear | MLP | FastKAN | **source effect** |
|---|---|---|---|---|
| VLM scores | ±  | ±  | ±  | |
| Expert labels | ±  | ±  | ±  | |
| **head effect** | | | | |

**Resampling, not seed-shuffling.** A linear head on 7 features has a convex objective — all seeds
converge to the same solution, std ≈ 0, and a paired test then measures initialisation noise rather
than anything real. Use repeated stratified k-fold over train+valid with the official test held out.

**Protocol written down before the first fit:** resampling scheme, metric, paired test named in
advance, test split touched once.

**Scale reference.** DermFM-Zero reports concept-source swings around 0.05 AUROC (CLIP 0.707 vs
0.757; MONET 0.765 vs 0.811). The head gap will likely be smaller. If the row gap dwarfs the column
gap, H3 lands.

---

### Step 5 — H2′ / H4: leakage and faithfulness

| Source | Head | Bal. acc | NEC vs VLM | NEC vs expert | **Δ (H4)** |
|---|---|---|---|---|---|
| VLM | Linear | | | | |
| VLM | MLP | | | | |
| VLM | FastKAN | | | | |
| Expert | Linear | | | | |
| Expert | MLP | | | | |
| Expert | FastKAN | | | | |

- **H4** = the Δ column is larger across the VLM-trained block than the expert-trained block.
- **H2′** = Δ barely moves down each source block.
- Accuracy is a control column. We expect to replicate f-CBM's small accuracy cost for the KAN head.

**NEC comes from the source equation, not from the name.** Pull it from VLG-CBM, not from a citing
paper's paraphrase. Whoever implements it gets a second person to verify it against a hand-computed
toy example before it produces any number that goes in a table.

---

## Why the KAN head, and how it is evaluated

An MLP puts fixed activations on nodes and learns weights on edges. A KAN inverts this: learnable
univariate spline functions sit on the edges. For a 7→2 head, every concept-to-logit relationship
is a readable curve φᵢⱼ(cᵢ) rather than a scalar.

A linear head says *pigment network: +2.3*. A KAN says whether the relationship is monotone,
saturating, or thresholded. For a faithfulness paper the shape is the relevant object — a threshold
near the clinical decision boundary is evidence the head learned dermatology; a non-monotone wiggle
is evidence it is fitting noise through the bottleneck.

**Evaluation axes — accuracy is the least important one:**

| Axis | Question | Where it lives |
|------|----------|----------------|
| Task accuracy | Does the bottleneck cost prediction? | Step 5 control column |
| Concept RMSE / AUROC | Are concepts detected correctly? | Step 2 (upstream of the head) |
| Leakage / NEC | Do concepts smuggle task information? | Step 5 — **the real head comparison** |
| Spline readability | Do curves match clinical priors? | Qualitative figure |

f-CBM's own ablation reports accuracy, concept RMSE and leakage together, and their generalisation
result is that adding their components to other CBMs substantially reduces leakage while costing a
small amount of accuracy. That trade *is* the claim.

**The spline-readability check is free.** Compare each KAN spline's sign and monotonicity against
the linear head's coefficient. MONET runs the linear version of this and finds the coefficients
recover clinical knowledge — ABCDE-matching concepts take positive weights, "blue" positive
(blue-white veils), "regular" negative. If our KAN curves reproduce that structure we have an
interpretability result at zero compute cost. If they do not while accuracy ties, that is a finding
about flexibility without grounding.

**One-sentence answer to a reviewer asking "why KAN":** the architecture we are responding to
proposes it as a leakage-mitigation device, its ablation was run on multimodal news data at a wide
bottleneck, and we test whether that benefit survives at a 7-concept medical bottleneck with
correlated expert-graded criteria.

---

## Paper map

Read in this order. Column three is the only thing needed from most of them.

| Step | Paper | What to take |
|---|---|---|
| 1 | MONET — Kim et al., *Nature Medicine* 2024 (also medRxiv 2023.06.07.23291119), Methods §"Automatic concept generation" | Eq. 1 scoring, reference-prompt normalisation, modality-specific templates |
| 1, 3 | Patrício et al., *Towards Concept-based Interpretability of Skin Lesion Diagnosis using VLMs* — arXiv 2311.14339 | Linear-probe protocol (penultimate layer, sklearn L-BFGS, 1000 iters); segmentation caveat |
| 2 | MONET Supplementary Table 4 | Per-concept term lists — the format for synonym ensembling |
| 2 | XCoOp — arXiv 2403.09410, §2.1 | Building clinical prompts *from* dataset concept annotations |
| 2 | Kawahara et al. 2018 (Derm7pt dataset paper) | Exact level names per criterion; our binarisation must match |
| 2 | DermFM-Zero — arXiv 2602.10624, Extended Data Table 18 | Reference AUROCs (pigment network: CLIP ~0.52, MONET ~0.80) |
| 2, 5 | Patrício et al., two-step — CSBJ 2025 / arXiv 2411.05609 | Ontology transfer: one concept vocabulary across all backbones |
| 3 | MONET Fig. 5F–G | The baseline set to compare against |
| 4 | Koh et al., *Concept Bottleneck Models*, ICML 2020 | The x→c→y definition; cite for "the head is stage 2" |
| 4 | PanDerm — arXiv 2410.15038 | Counter-evidence on derm-pretrained vs general CLIP |
| 5 | **f-CBM — arXiv 2603.13163**, Fig. 4 + Table 2 | Leakage definition, the KAN's actual claim, the accuracy cost we replicate |
| 5 | VLG-CBM — arXiv 2408.01432 | NEC's originating equation |
| 5 | MONET Fig. 5H–I | Precedent for checking learned weights against clinical knowledge |

**f-CBM is the only must-read-in-full.** All of Step 5 depends on their leakage definition.

---

## Rules

1. **Nothing gets fine-tuned.** Frozen backbones, cached features, probes on top. The moment
   someone starts fine-tuning a ViT, we lose the semester.
2. **Expert labels are the ground truth. Always.** We never evaluate VLM concept scores against
   VLM-derived labels. That is the exact circularity we are criticising.
3. **10 seeds per cell, mean ± std, paired test.** Effect sizes here are small. Single-run numbers
   are worthless.
4. **NEC comes from the equation in the source paper, not from the name.** Second person verifies
   against a hand-computed toy example first.
5. **The results freeze holds.** No new experiments once writing starts.
6. **If we slip, cut in this order:** third backbone → robustness → PH2 transfer. Never the
   expert-label comparison. Never the seeds.

---

## Stated limitations (write these in methods; do not let a reviewer find them first)

- **Single dataset in V1.** ~1,000 images, one institution. The H1 spread finding is about this
  corpus; whether prompt sensitivity has the same shape elsewhere is untested until PH2.
- **Unsegmented images.** Patrício et al. segment lesions before scoring and report it improves
  results. Our absolute AUROCs are therefore not directly comparable to theirs.
- **Fairness** is stated as a dataset limitation, not measured. No public dermoscopic dataset has
  diverse skin tone coverage.
- **Phase 1 correction.** Our earlier claim that dermatology has no concept-level annotations was
  wrong. Derm7pt (1,011 images, 7-point checklist), PH2 (200 images, 5 features + ABC + masks) and
  SkinCon (3,230 images, 48 concepts) all exist. Expert concept labels exist but only at 1–3k
  scale — enough to audit automatic labels, not enough to train on. That sentence is the paper's
  motivation.

---

## Decisions to freeze before Step 4

- [ ] Diagnosis target binarisation: melanoma vs all, or melanoma vs nevus with miscellaneous
      classes dropped. The second mirrors MONET's "well-defined clinical task" convention and is
      cleaner, but shrinks n. Pick one, write it into `prompt_protocol.json`, do not revisit.
- [ ] λ value (default 1)
- [ ] Resampling scheme and paired test for Step 4
- [ ] NEC implementation verified by second person

---

## Environment

Kaggle T4 is sufficient for the entire project — under an hour of GPU time end to end. Do not buy
compute.

- Encode pass: 1–2 min
- All 12 H1 cells: seconds (cached matmuls, CPU)
- All head fits: seconds (sklearn, 7-dim input)

`/kaggle/working` does not survive session restarts. Save `img_emb.npy` and `img_idx.npy` as a
Kaggle Dataset after the first encode, then iterate on prompts in a CPU session with no Internet
toggle needed. Notebook Internet must be ON for `pip install open_clip_torch`; phone verification
may be required to enable it — check this before you need it.

---

## Repo layout

```
data/           # loaders, binarisation, splits
prompts/        # the 12-cell grid, templates, term lists
encode/         # one-shot image + text encoding, caching
score/          # MONET Eq. 1 / Eq. 2 scoring
heads/          # linear, MLP, FastKAN
eval/           # AUROC + bootstrap, NEC, baselines
protocol/       # frozen JSON: prompts, λ, selection rule, binarisations
results/        # tables, one file per step
```