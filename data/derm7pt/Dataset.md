# DERM7PT

The seven point checklist dermoscopy dataset.

## Citations

Dataset paper:

```
@article{Kawahara2018-7pt,
  author = {Kawahara, Jeremy and Daneshvar, Sara and Argenziano, Giuseppe and Hamarneh, Ghassan},
  doi = {10.1109/JBHI.2018.2824327},
  issn = {2168-2194},
  journal = {IEEE Journal of Biomedical and Health Informatics},
  month = {mar},
  number = {2},
  pages = {538--546},
  publisher = {IEEE},
  title = {Seven-point checklist and skin lesion classification using multitask multimodal neural nets},
  volume = {23},
  year = {2019}
}
```

Consistency analysis, and the source of every statistic below:

Napoles, Grau, Salgueiro. *Concept Inconsistency in Dermoscopic Concept Bottleneck Models: A Rough-Set Analysis of the Derm7pt Dataset.* arXiv:2604.19323.
Cleaned dataset and code: https://github.com/gnapoles/Consistent-Derm7pt

## Dataset facts

| Quantity | Value |
|---|---|
| Images | 1,011 |
| Melanoma | 252 (24.9%) |
| Class ratio (melanoma to non melanoma) | 1:3.0 |
| Majority class baseline | 75.1% |
| Unique concept profiles (full multi valued concepts) | 305 |
| Inconsistent profiles | 50 (16.4%) |
| Images in the boundary region | 306 (30.3%) |
| Quality of classification, gamma | 0.697 |
| Hard CBM accuracy ceiling | 92.1% |
| Melanoma inside the boundary region | 136 of 252 (54.0%) |

Our loader drops 9 cases where the clinic and derm columns point at the same file, so our working n is 1,002. Counts will not match the paper exactly. State this once in methods.

## Per concept positives under our binarisation

Three questions per concept. What does the sign mean clinically. How many lesions carry the positive levels. Among lesions carrying that level, how often was the diagnosis melanoma.

| Concept | Clinical meaning | Our positive levels | n positive | % of 1,011 | Melanoma prevalence |
|---|---|---|---|---|---|
| Dots and globules | Small dots or larger round or oval pigment structures | irregular | 448 | 44.3% | 49% |
| Pigmentation | How dark pigment is distributed within the lesion | diffuse irregular, localized irregular | 305 | 30.2% | 48%, 35% |
| Regression structures | Areas suggesting previous loss of lesion tissue | blue areas, white areas, combinations | 253 | 25.0% | 30%, 39%, 61% |
| Streaks | Radial streaks or pseudopods extending toward the edge | irregular | 251 | 24.8% | 61% |
| Pigment network | Brown network of pigmented lines around lighter holes | atypical | 230 | 22.7% | 60% |
| Blue whitish veil | Structureless blue white area over part of the lesion | present | 195 | 19.3% | 62% |
| Vascular structures | Visible blood vessel patterns | linear irregular, dotted | 71 | 7.0% | 94%, 55% |

Dataset base rate for comparison: 24.9%.

### 1. Dots and globules

Pigment deposits. Regular is 0, irregular is 1.

448 lesions carry irregular dots and globules, which is 44.3% of the dataset. This is the most common positive concept but it is not a majority. Melanoma prevalence among them is 49%, roughly double the base rate.

### 2. Pigmentation

How pigment is distributed across the lesion. Two positive levels.

* Diffuse irregular: irregular pigmentation spread across the lesion. 48% melanoma prevalence.
* Localized irregular: irregular pigmentation concentrated in one region. 35% melanoma prevalence.

Together 305 lesions, 30.2%.

Note that localized regular has only n equals 3 in the whole dataset, so that level is effectively empty and contributes nothing to the negative class.

### 3. Regression structures

Parts of the lesion that have undergone loss of previously pigmented or cellular tissue.

Blue areas, white areas, and combinations are all coded 1. Absent is 0. This binarisation is clinical, taken from the checklist definition, not derived from the prevalence numbers.

The prevalences are worth recording because they are heterogeneous:

* blue areas: 30%
* white areas: 39%
* combinations: 61%

Blue areas alone sit at 30% against a base rate of 24.9%, which is barely above chance, while combinations sit at 61%. So our positive class for this concept mixes a nearly uninformative level with a strongly predictive one. If regression structures scores poorly in H1, this is the most likely reason, and it deserves a per level breakdown rather than being left as a mystery.

### 4. Streaks

Radial structures extending toward the lesion edge. Regular is 0, irregular is 1.

251 lesions, 24.8%, with 61% melanoma prevalence. Irregular streaks are among the strongest single indicators in the dataset.

### 5. Pigment network

Network of pigmented lines. Typical is 0, atypical is 1. Absent also falls in the negative class under our atypical versus rest binarisation.

230 lesions, 22.7%, with 60% melanoma prevalence.

### 6. Blue whitish veil

Blue white or blue gray opaque area over part of the lesion. Absent is 0, present is 1.

195 lesions, 19.3%, with 62% melanoma prevalence. This is the highest prevalence of any binary concept in our set.

### 7. Vascular structures

Blood vessels visible under dermoscopy. Linear irregular and dotted are coded 1.

* linear irregular: 94% melanoma prevalence, n equals 18
* dotted: 55% melanoma prevalence, n equals 53

Together only 71 lesions, 7.0%.

One deliberate exclusion to declare: the level "within regression" (n equals 46) carries 61% melanoma prevalence and sits in our negative class. Argenziano's checklist defines atypical vessels as linear irregular or dotted, so the exclusion is defensible, but a high melanoma level sitting in the negative class must be stated in one sentence in methods rather than left implicit.

## Vascular structures is the problem child

71 positives across the whole dataset means roughly 14 in the validation split. Our evaluate guard only skips a concept when positives fall below 2, so it will happily return an AUROC with a bootstrap interval so wide it carries no information. The danger is not a wrong number, it is that the number would feed the mean used to select the winning prompt strategy.

Decision, written into the protocol before any run:

* Vascular structures is excluded from the mean AUROC used for prompt cell selection.
* It is reported separately on the pooled set of all splits, n positive equals 71, explicitly flagged as non independent because pooling breaks the split discipline.
* The pooled result is a caveat analysis. It never feeds selection, and it never appears in the headline table without the flag.

## Metric choice

Melanoma is a 1:3 minority, so the Step 3 floor row is 0.751, not 0.500.

In binary classification micro F1 equals accuracy, which is exactly the metric that misleads under this imbalance. The consistency paper uses macro averaged F1 for this reason.

Primary metric: macro averaged F1, or balanced accuracy. Report plain accuracy alongside so the comparison against the 75.1% majority baseline stays visible.

## Our accuracy ceiling is lower than 92.1%

The theorem in the paper bounds any classifier that is a deterministic function of the hard concept vector at 92.1% accuracy. Our expert label arm is exactly that: seven binary values, then a head, then a diagnosis.

But 92.1% is computed over 305 profiles built from the full multi valued concepts. We collapse to seven binary values, so at most 128 distinct profiles. More collapse means more images sharing a signature, which means more inconsistency, which means a strictly lower ceiling.

Task for day one, before any experiment:

```python
key = concepts[CRITERIA]
grp = key.assign(mel=labels).groupby(CRITERIA)['mel'].agg(['size', 'sum'])
correct = grp.apply(lambda r: max(r['sum'], r['size'] - r['sum']), axis=1).sum()
ceiling = correct / len(labels)
```

This is a groupby, not an experiment. Ten minutes. The number goes in methods and is the most defensible statement in the paper, because it is a theorem rather than a measurement.

### What the ceiling buys us

Add an "excess over ceiling" column to the Step 5 table.

The expert label arm uses hard concepts, so the theorem binds it. The VLM arm uses soft continuous scores, so the theorem does not bind it directly, and that asymmetry is the point: soft scores can carry information a hard vector cannot.

If the VLM arm sits above the hard ceiling while the expert arm does not, that is a demonstration of leakage that a reviewer can verify by hand, independent of whether our NEC implementation is correct. It is a free second opinion on H4.

## H1a, registered in advance

The concepts our VLM is most likely to fail on are the concepts human annotators disagree on. The boundary region is concentrated in exactly the concepts we score:

| Concept level | In boundary profiles | In consistent profiles |
|---|---|---|
| Irregular dots and globules | 79% | 29% |
| Irregular streaks | 39% | 19% |
| Blue whitish veil present | 29% | 15% |

So a poor VLM score has two possible causes: the model genuinely cannot recognise the structure, or the concept itself is ambiguous to dermatologists. We can partially separate them by reporting H1 agreement on consistent profile images and boundary region images separately.

Registered as a hypothesis, not kept as a fallback explanation:

> **H1a.** VLM to expert agreement is lower on boundary region images than on consistent profile images, indicating that measured disagreement partly reflects genuine perceptual ambiguity rather than model failure.

If agreement is uniformly low across both groups, the cause is the model. If it collapses only on boundary images, the VLM is tracking the same ambiguity dermatologists do, which is the more interesting result. Either outcome is reportable, and registering it in advance stops it reading as an excuse written after the numbers came back.

## Supervised reference numbers

Across 19 CNN backbones trained with full expert concept supervision (EfficientNet, DenseNet, ResNet, Wide ResNet families), concept accuracy tops out at 0.70. The authors conclude that concept annotation noise rather than backbone capacity is the binding constraint on bottleneck quality.

That is an independent group reaching our H3 conclusion from the opposite direction. They varied the backbone across 19 architectures and found it did not matter. Cite it in the motivation.

It also gives us a supervised reference point. If zero shot CLIP lands anywhere near 0.70 concept accuracy, the gap between free VLM labels and training on expert labels is smaller than the field assumes.

Best label performance under symmetric filtering: efficientnet_b5, label F1 0.85, label accuracy 0.90, concept accuracy 0.70.

## Derm7pt+ variants

| Property | No filter | Asymmetric | Symmetric |
|---|---|---|---|
| Images | 1,011 | 841 | 705 |
| Melanoma retained | 252 | 252 | 116 |
| Class ratio | 1:3.0 | 1:2.3 | 1:5.1 |
| Quality of classification | 0.697 | above 0.697 | 1.000 |
| Accuracy ceiling | 92.1% | 83.2% | 100% |
| Concept consistency | none | partial | full |

The released Derm7pt+ images are also manually cropped to 384 by 384 with border artifacts removed.

## Experiment order

**Confound to avoid.** Derm7pt+ changes two things at once, filtering and cropping. A straight comparison of Derm7pt against Derm7pt+ cannot attribute any difference to one or the other.

1. **Crop ablation first, early, on the unfiltered set, one prompt cell.** Raw against cropped, twenty minutes. This de confounds everything downstream and costs almost nothing.
2. **Run H1 on the original unfiltered Derm7pt.** This is the primary result. The ambiguous cases are the interesting ones and filtering them away removes the phenomenon we are studying.
3. **Report H1 split by boundary region against consistent profiles.** This is H1a.
4. **Re run on Derm7pt+ as a robustness appendix.** With the crop ablation already done, this becomes a clean filtering only comparison. Question: does the H1 prompt spread change on the consistent subset.
5. **Raw against cropped for the best head and best prompt cell** only after the above, as a final confirmation rather than a discovery step.

## Corrections log

Fixed from the first draft of these notes, kept so the same errors do not return:

* Micro F1 replaced by macro F1. Micro F1 equals accuracy in the binary case and is the metric the imbalance breaks.
* Dots and globules is 44.3%, not more than half.
* Denominator is 1,011, not 1,100. Our working n after dropping duplicates is 1,002.
* Regression structures binarisation is clinical, taken from the checklist. It is not justified by combinations scoring higher than white areas alone.
* Vascular positives are 71, not 70. Pooled evaluation is a flagged caveat, never an input to selection.