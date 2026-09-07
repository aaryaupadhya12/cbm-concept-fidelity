"""
protocol.py -- the frozen audit protocol, and the contract a new dataset must meet.

WHY THIS FILE EXISTS
--------------------
The goal is that a teammate can point this at PH2, or at a knee X-ray dataset,
and get a comparable answer without re-deriving anything. That only works if
exactly one thing varies between runs: the dataset adapter. Everything else --
null construction, p-value formula, correction, detection floor -- is frozen.

THE FAILURE MODE THIS PREVENTS
------------------------------
Your teammates will use coding agents. Agents are very good at making code run
and very bad at leaving a statistical protocol alone. Asked to "adapt the audit
to PH2", an agent will cheerfully widen the null, drop the +1 in the p-value,
change the binarisation because a concept had few positives, or switch AUROC to
accuracy because AUROC crashed on a degenerate split. Each of those silently
makes the result incomparable to yours, and none of them will look like an error.

So: this module is IMPORTED, never edited. Teammates write an adapter and
nothing else. Every result file carries PROTOCOL_HASH. Two runs whose hashes
differ are not comparable, and the comparison table should refuse to merge them.

WHAT A TEAMMATE ACTUALLY WRITES
-------------------------------
One class, five methods, ~60 lines. See PH2Adapter below for the shape.
"""

import hashlib
import inspect
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# FROZEN PROTOCOL CONSTANTS -- changing any of these invalidates comparability
# --------------------------------------------------------------------------

# The object of study is the CONCEPT BOTTLENECK MODEL, not the VLM. Concept
# agreement (H1) is the instrument. A dataset run that reports only H1 is half an
# experiment and must not enter the cross-dataset table.
REQUIRED_ARMS = ("expert", "vlm", "vlm_shuffled", "vlm_continuous")

PROTOCOL = dict(
    version="1.1",
    required_arms=list(REQUIRED_ARMS),
    n_null_draws=1000,
    p_formula="(beaten + 1) / (draws + 1)",
    correction="bonferroni",
    primary_concept_metric="auroc",
    primary_diagnosis_metric="macro_f1",
    # After dataset 1, the prompt grid is FROZEN to a single cell. See note below.
    neg_mode="antonym",
    ensemble=True,
    min_positives=10,      # below this a concept is reported, never used in selection
)

PROTOCOL_HASH = hashlib.sha256(
    json.dumps(PROTOCOL, sort_keys=True).encode()).hexdigest()[:12]


# --------------------------------------------------------------------------
# WHY THE GRID FREEZES AFTER DATASET 1
# --------------------------------------------------------------------------
# On Derm7pt you ran 6 cells x 7 concepts = 42 tests. That was exploratory and it
# is defensible as dataset 1: you did not know which prompt construction was
# reasonable. If you repeat the full grid on every dataset the burden multiplies:
#
#   3 datasets x 6 cells x 7 concepts       = 126 tests, alpha = 4.0e-4
#   3 datasets x 1 cell  x 7 concepts       =  21 tests, alpha = 2.4e-3
#
# The second is a real experiment. The first is a fishing expedition wearing a
# correction. Datasets 2 and 3 run ONE pre-registered cell. The Derm7pt grid
# becomes a methods section justifying that choice, not a template to repeat.


# --------------------------------------------------------------------------
# THE CONTRACT
# --------------------------------------------------------------------------

@dataclass
class DatasetSpec:
    """Everything the protocol needs to know about a dataset. Nothing else."""
    name: str
    modality: str                    # "dermoscopy", "clinical photo", "radiograph"
    n_images: int
    concepts: List[str]
    binarisation: Dict[str, str]     # concept -> plain-English rule, for the paper
    diagnosis_task: str              # "melanoma vs rest", "KL>=2 vs rest"
    splits: Dict[str, int]           # split name -> n
    source: str                      # citation / URL


class DatasetAdapter:
    """Subclass this. Implement five methods. Touch nothing else in the repo.

    The protocol calls these in a fixed order and does not care how you
    implement them, only that the shapes and dtypes match.
    """

    def spec(self) -> DatasetSpec:
        raise NotImplementedError

    def image_paths(self) -> Sequence[str]:
        """One path per image, in a fixed order. This order defines every index."""
        raise NotImplementedError

    def concept_labels(self) -> pd.DataFrame:
        """Binary 0/1, one column per concept, rows aligned to image_paths().

        Binarisation must come from the clinical definition, not from the data.
        Do NOT pick thresholds to balance classes -- that is fitting the label to
        the result. Record the rule in spec().binarisation as plain English.
        """
        raise NotImplementedError

    def diagnosis(self) -> np.ndarray:
        """Binary 0/1 diagnosis, aligned to image_paths()."""
        raise NotImplementedError

    def split_mask(self, name: str) -> np.ndarray:
        """Boolean mask over image_paths(). Must include 'test'."""
        raise NotImplementedError

    # -- provided, do not override --

    def validate(self) -> None:
        sp = self.spec()
        paths, C, y = self.image_paths(), self.concept_labels(), self.diagnosis()
        assert len(paths) == len(C) == len(y) == sp.n_images, "length mismatch"
        assert list(C.columns) == sp.concepts, "concept order must match spec"
        assert set(np.unique(C.values)) <= {0, 1}, "concept labels must be 0/1"
        assert set(np.unique(y)) <= {0, 1}, "diagnosis must be 0/1"
        assert set(sp.binarisation) == set(sp.concepts), "document every binarisation"
        masks = {k: self.split_mask(k) for k in sp.splits}
        assert "test" in masks, "a held-out test split is mandatory"
        stacked = np.vstack(list(masks.values()))
        assert stacked.sum(0).max() <= 1, "splits overlap -- leakage"
        thin = [c for c in sp.concepts if C[c].sum() < PROTOCOL["min_positives"]]
        if thin:
            print(f"  NOTE: {thin} have <{PROTOCOL['min_positives']} positives. "
                  "They are reported but excluded from any selection mean.")
        print(f"  adapter ok: {sp.name}, n={sp.n_images}, "
              f"{len(sp.concepts)} concepts, protocol {PROTOCOL_HASH}")


# --------------------------------------------------------------------------
# THE NUMBER THAT MAKES CROSS-DATASET COMPARISON HONEST
# --------------------------------------------------------------------------

def detection_floor(null_runs_for_concept, q=95):
    """The 95th percentile of the null IS the minimum detectable AUROC.

    This is the single most important number for going multi-dataset, and it is
    free -- you already compute the null.

    Why it matters: PH2 has 200 images. Derm7pt's test split has ~395. A smaller
    evaluation set makes the null WIDER, so random text reaches higher AUROCs by
    luck, so the bar for a real result rises. If you report raw AUROCs across
    datasets without this, a 0.62 on PH2 and a 0.62 on Derm7pt look equal when
    one may be below its own noise floor and the other above it.

    Report per concept per dataset:
        auroc, floor, headroom = auroc - floor
    Headroom is comparable across datasets in a way raw AUROC is not.
    """
    return float(np.percentile(np.asarray(null_runs_for_concept), q))


def floor_table(runs: Dict[str, list], scores_auroc: Dict[str, float]) -> pd.DataFrame:
    rows = []
    for c, r in runs.items():
        f = detection_floor(r)
        rows.append({"concept": c, "auroc": round(scores_auroc[c], 3),
                     "floor_p95": round(f, 3),
                     "headroom": round(scores_auroc[c] - f, 3),
                     "above_floor": scores_auroc[c] > f})
    return pd.DataFrame(rows).sort_values("headroom", ascending=False)


def assert_full_chain(cbm_results: pd.DataFrame) -> None:
    """A dataset contributes only if all four bottleneck arms were run.

    expert          -- hard concepts, bounded by the ceiling theorem
    vlm             -- binarised VLM scores, same representation as expert, so a
                       gap between them is attributable to concept SOURCE alone
    vlm_shuffled    -- marginals preserved, image-concept link destroyed
    vlm_continuous  -- soft scores, NOT bounded by the theorem

    Leaving out vlm_shuffled loses the control. Leaving out vlm_continuous loses
    the leakage proof. Either omission turns a CBM audit into a CLIP benchmark.
    """
    have = set(cbm_results["source"].unique())
    missing = set(REQUIRED_ARMS) - have
    if missing:
        raise ValueError(f"incomplete run: missing arms {sorted(missing)}. "
                         "Concept agreement alone does not support a CBM claim.")


def leakage_verdict(scores_by_arm: Dict[str, float], hard_ceiling: float,
                    shuffled_null=None, metric="macro_f1") -> str:
    """The one inference that is a theorem rather than an inference.

    scores_by_arm : arm -> value of the PRIMARY metric (macro-F1), not accuracy.
    shuffled_null : optional list of vlm_shuffled scores over several permutation
                    seeds. One permutation is a sample, not a control.

    Why the primary metric must be macro-F1 here: at Derm7pt's 1:3 imbalance an
    always-majority classifier scores 0.751 accuracy and 0.429 macro-F1. An arm
    sitting at 0.44 macro-F1 has learned nothing, but its accuracy still looks
    like 0.75 and can even beat another arm by a fraction of a percent. Comparing
    arms on accuracy manufactures differences out of a single reordered image.

    NOTE ON WHAT THIS DOES *NOT* PROVE. Poor per-concept agreement alone does not
    establish that the bottleneck carries non-concept information: several weakly
    tracked concepts can aggregate. Only ceiling excess settles it. Cite the
    ceiling first and the null second.
    """
    vlm = scores_by_arm.get("vlm", float("nan"))
    shuf = scores_by_arm.get("vlm_shuffled", float("nan"))
    cont = scores_by_arm.get("vlm_continuous", float("nan"))
    acc_cont = scores_by_arm.get("vlm_continuous_acc", cont)

    lines = []
    excess = acc_cont - hard_ceiling
    if excess > 0:
        lines.append(f"LEAKAGE PROVEN: continuous bottleneck exceeds the hard-concept "
                     f"ceiling by {excess:+.4f}. No assignment of the named concepts "
                     f"reaches this accuracy.")
    else:
        lines.append(f"NO CEILING EXCESS: continuous arm sits {excess:+.4f} below the "
                     f"hard-concept ceiling. Leakage is NOT proven.")

    if shuffled_null is not None and len(shuffled_null) >= 5:
        arr = np.asarray(shuffled_null, dtype=float)
        n = len(arr)
        p = (int((arr >= vlm).sum()) + 1) / (n + 1)
        floor_p = 1.0 / (n + 1)
        z = (vlm - arr.mean()) / (arr.std() + 1e-9)
        lines.append(f"vs shuffled control ({n} permutations): "
                     f"vlm={vlm:.4f}, shuffled={arr.mean():.4f}+/-{arr.std():.4f}, "
                     f"p={p:.4f}, effect={z:.1f} SD")
        # RESOLUTION TRAP -- the same one that bit the null draws. With n
        # permutations the smallest observable p is 1/(n+1). At n=10 that is
        # 0.0909, so "p < 0.05" is UNREACHABLE no matter how large the effect.
        # A 10-SD separation would still be reported as non-significant.
        # So: judge on effect size when p is at its floor, and say why.
        if floor_p >= 0.05:
            lines.append(f"  NOTE: {n} permutations cannot produce p<0.05 "
                         f"(floor is {floor_p:.4f}). Judging on effect size instead. "
                         f"Rerun with >=20 permutations for a usable p-value.")
            carries = z > 3.0
        else:
            carries = p < 0.05
    else:
        delta = vlm - shuf
        lines.append(f"vs shuffled control (SINGLE permutation -- run >=10 seeds): "
                     f"delta={delta:+.4f} on {metric}")
        carries = delta > 0.05

    if not carries:
        lines.append("VERDICT: the VLM bottleneck is INDISTINGUISHABLE from its "
                     "shuffled control. The seven dimensions carry no more diagnostic "
                     "information than randomly permuted values wearing the same "
                     "clinical names. Report this directly -- it is the strongest "
                     "form of the claim, not a weak one.")
    else:
        lines.append("VERDICT: the VLM bottleneck beats its shuffled control but stays "
                     "under the ceiling. Consistent with weak concept signal "
                     "aggregating. Do NOT call this leakage.")
    return "\n".join(lines)


def stamp(result_df: pd.DataFrame, spec: DatasetSpec, backbone: str) -> pd.DataFrame:
    """Every result row carries what produced it. Merging across hashes is a bug."""
    out = result_df.copy()
    out["dataset"] = spec.name
    out["modality"] = spec.modality
    out["backbone"] = backbone
    out["protocol_hash"] = PROTOCOL_HASH
    return out


def assert_comparable(frames: Sequence[pd.DataFrame]) -> None:
    hashes = {h for f in frames for h in f["protocol_hash"].unique()}
    if len(hashes) > 1:
        raise ValueError(
            f"refusing to merge results from different protocols: {sorted(hashes)}. "
            "Someone edited protocol.py. Find out what changed before comparing.")


# --------------------------------------------------------------------------
# WORKED EXAMPLE -- the shape a teammate copies
# --------------------------------------------------------------------------

class PH2Adapter(DatasetAdapter):
    """PH2: 200 dermoscopic images, ABCD + 5 dermoscopic criteria, lesion masks.

    NOTE FOR WHOEVER WRITES THIS FOR REAL. PH2's concepts are ABCD-style GLOBAL
    lesion properties (asymmetry, border regularity, colour count) rather than
    the LOCAL microstructures in Derm7pt. That is not a nuisance, it is the most
    interesting prediction the project makes: a general VLM should plausibly do
    better on "the lesion is asymmetric" than on "the pigment network is
    atypical", because the former is describable in ordinary visual language.
    If that holds, you have a rule for WHICH concepts VLM scoring can be trusted
    on, which is far more useful than a per-dataset verdict.

    It is also why the OpenCV/ABC baseline earns its place here: on PH2 it is a
    direct competitor, not a courtesy comparison.
    """

    def spec(self):
        return DatasetSpec(
            name="PH2", modality="dermoscopy", n_images=200,
            concepts=["asymmetry", "irregular_border", "multiple_colours",
                      "atypical_network", "blue_whitish_veil"],
            binarisation={
                "asymmetry": "PH2 asymmetry grade >= 1 (asymmetric in >=1 axis)",
                "irregular_border": "border irregularity present",
                "multiple_colours": ">= 3 colours recorded",
                "atypical_network": "atypical pigment network present",
                "blue_whitish_veil": "blue-whitish veil present",
            },
            diagnosis_task="melanoma vs common+atypical nevus",
            splits={"test": 200},   # PH2 has no standard split; whole set is test
            source="Mendonca et al., PH2 dermoscopic image database, EMBC 2013",
        )

    def image_paths(self):
        raise NotImplementedError("fill in")

    def concept_labels(self):
        raise NotImplementedError("fill in")

    def diagnosis(self):
        raise NotImplementedError("fill in")

    def split_mask(self, name):
        raise NotImplementedError("fill in")


if __name__ == "__main__":
    print(f"protocol {PROTOCOL['version']}  hash {PROTOCOL_HASH}")
    print(json.dumps(PROTOCOL, indent=2))
    print("\nPH2 spec (concepts a teammate must binarise and document):")
    print(json.dumps(asdict(PH2Adapter().spec()), indent=2))
