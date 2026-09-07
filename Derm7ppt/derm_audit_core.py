"""
derm_audit_core.py -- corrected scoring core for the H1 concept audit.

Replaces concept_scores / null_runs / evaluate_vs_null from the notebook.
Everything else in your notebook (load_meta, encode_images, hard_ceiling) stays.

WHAT EACH PIECE IS, IN ONE LINE EACH
------------------------------------
encode_texts   turn a sentence into a 768-number vector
_vec           turn a phrase into ONE unit vector, optionally averaged over templates
concept_scores for each image, a number in [0,1] saying "how present is this concept"
null_runs      the same machinery fed nonsense, 200 times, to see how high luck gets
evaluate       compare real score against the null, with honest multiple-comparison maths

Run `python derm_audit_core.py` with no CLIP installed to execute the selftest,
which drives the whole pipeline with a fake encoder whose answers are known.
"""

import itertools
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

CRITERIA = ["pigment_network", "blue_whitish_veil", "vascular_structures",
            "pigmentation", "streaks", "dots_and_globules",
            "regression_structures"]

LAMBDA = 1.0

# --------------------------------------------------------------------------
# FIX 1 -- reference prompt was a truncated phrase
# --------------------------------------------------------------------------
# Was: REF_TEMPLATE = "dermoscopy of"
# CLIP encodes that dangling "of" as if it were a real sentence, and you get a
# vector for a grammatical fragment rather than for the idea "a dermoscopy image".
# The reference is supposed to mean "a generic image of this kind, no concept" --
# so it has to be a complete phrase.
REFERENCE = "a dermoscopic image of a skin lesion"

TEMPLATES = [
    "{}",
    "dermoscopy of {}",
    "dermatoscopy of {}",
    "a dermoscopic image of {}",
]

SHORT = {
    "pigment_network":       ("atypical pigment network", ["typical pigment network"]),
    "blue_whitish_veil":     ("blue-whitish veil", ["no blue-whitish veil"]),
    "vascular_structures":   ("atypical vessels", ["regular vessels"]),
    "pigmentation":          ("irregular pigmentation", ["regular pigmentation"]),
    "streaks":               ("irregular streaks", ["regular streaks"]),
    "dots_and_globules":     ("irregular dots and globules", ["regular dots and globules"]),
    "regression_structures": ("regression structures", ["no regression structures"]),
}

ANTONYM_SET = {
    "pigment_network":       ["typical pigment network", "regular thin network lines",
                              "no pigment network"],
    "blue_whitish_veil":     ["no blue-whitish veil", "uniform brown pigmentation"],
    "vascular_structures":   ["comma vessels", "arborizing vessels", "no visible vessels"],
    "pigmentation":          ["regular pigmentation", "evenly distributed pigment"],
    "streaks":               ["regular streaks", "symmetric radial streaks", "no streaks"],
    "dots_and_globules":     ["regular dots and globules", "uniform globules",
                              "no dots or globules"],
    "regression_structures": ["no regression structures", "intact pigmented lesion"],
}


def load_wordings(json_path=None):
    """Build the three wording tiers. Tier A is hardcoded; B and C come from JSON.

    FIX 2 -- your notebook calls concept_scores(wording="clinical") but CLINICAL
    is never defined anywhere. That is a NameError waiting to happen the moment
    you leave wording="short".
    """
    wordings = {"short": SHORT}
    if json_path and Path(json_path).exists():
        p = json.load(open(json_path))["concepts"]
        wordings["taxonomy"] = {c: (p[c]["B"]["pos"], [p[c]["B"]["neg"]]) for c in CRITERIA}
        wordings["clinical"] = {c: (p[c]["C"]["pos"], [p[c]["C"]["neg"]]) for c in CRITERIA}
    return wordings


# --------------------------------------------------------------------------
# TEXT ENCODING
# --------------------------------------------------------------------------

_ENCODER = None       # set by set_encoder(); the selftest injects a fake one


def set_encoder(fn):
    """fn(list_of_strings) -> (n, d) array of L2-normalised row vectors."""
    global _ENCODER
    _ENCODER = fn


def encode_texts(strings):
    if _ENCODER is None:
        raise RuntimeError("call set_encoder(...) first")
    v = np.asarray(_ENCODER(list(strings)), dtype=np.float64)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


_VEC_CACHE = {}


def clear_vec_cache():
    _VEC_CACHE.clear()


def _vec(phrase, ensemble):
    """One unit vector per phrase.

    ensemble=True averages the phrase across the 4 templates then re-normalises.
    Averaging cancels template-specific noise; it is the standard CLIP trick.
    """
    # FIX 7 -- memoise. The same nonsense phrase is re-encoded once per draw per
    # concept; with 1000 draws x 7 concepts that is tens of thousands of
    # redundant forward passes. Caching makes 1000 draws about as fast as 200.
    key = (phrase, ensemble)
    if key in _VEC_CACHE:
        return _VEC_CACHE[key]
    if not ensemble:
        v = encode_texts([phrase])[0]
    else:
        v = encode_texts([t.format(phrase) for t in TEMPLATES]).mean(0)
    v = v / np.linalg.norm(v)
    _VEC_CACHE[key] = v
    return v


# --------------------------------------------------------------------------
# SCORING
# --------------------------------------------------------------------------

def _softmax_score(img_emb, pos_v, neg_vs):
    """P(concept) = exp(sim_pos) / (exp(sim_pos) + sum_j exp(sim_neg_j)).

    img_emb @ v is the cosine similarity of every image to that text vector,
    because both are unit-length. So this is one number per image per prompt.
    """
    p = np.exp((img_emb @ pos_v) / LAMBDA)
    q = sum(np.exp((img_emb @ n) / LAMBDA) for n in neg_vs)
    return p / (p + q)


def negatives_for(concept, neg_mode, wording_table, ensemble):
    """FIX 3 -- ensembling was applied to the positive but not to the reference.

    In your notebook, neg_mode='ref' called encode_texts([REF_TEMPLATE]) directly,
    so with ensemble=True the positive was template-averaged and the negative was
    not. Any difference between those cells then mixes 'ensembling helps' with
    'the two sides were built differently', which is not a thing you can untangle
    afterwards. Both sides now go through _vec with the same flag.
    """
    if neg_mode == "ref":
        return [_vec(REFERENCE, ensemble)]
    if neg_mode == "antonym":
        return [_vec(wording_table[concept][1][0], ensemble)]
    if neg_mode == "antonym_set":
        return [_vec(a, ensemble) for a in ANTONYM_SET[concept]]
    raise ValueError(neg_mode)


def concept_scores(img_emb, wording_table, neg_mode="ref", ensemble=False):
    out = {}
    for c in CRITERIA:
        pos_v = _vec(wording_table[c][0], ensemble)
        out[c] = _softmax_score(img_emb, pos_v, negatives_for(c, neg_mode, wording_table, ensemble))
    return out


def n_negatives(concept, neg_mode, wording_table):
    return len(negatives_for.__wrapped__(concept)) if False else (
        1 if neg_mode in ("ref", "antonym") else len(ANTONYM_SET[concept])
    )


# --------------------------------------------------------------------------
# NULL
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# FIX 6 -- the null vocabulary was too small to resolve the p-values you want
# --------------------------------------------------------------------------
# In 'ref' mode the negative slot is FIXED (the reference phrase), so only the
# positive varies. With 22 object names there are only 22 distinct null values,
# no matter how many draws you request -- p resolution is 1/22 = 0.045, and
# raising n_draws to 1000 changes nothing. Adjective x noun gives thousands of
# distinct phrases, so the number of draws becomes the real limit again.
ADJECTIVES = ["red", "wooden", "small", "broken", "shiny", "old", "plastic",
              "blue", "heavy", "narrow", "empty", "folded", "rusty", "bright",
              "green", "smooth", "tall", "worn", "metal", "round",
              "painted", "hollow", "striped", "plain", "cracked", "wide",
              "yellow", "soft", "curved", "flat", "spare", "loose"]
NOUNS = ["bicycle", "chair", "banana", "keyboard", "elephant", "lamp",
         "mountain", "spoon", "violin", "toaster", "bridge", "sock",
         "teapot", "helicopter", "cactus", "stapler", "lighthouse",
         "mushroom", "bucket", "giraffe", "candle", "tractor", "pencil",
         "waterfall", "drum", "envelope", "ladder", "pineapple", "kettle",
         "harbour", "saddle", "lantern", "trumpet", "anchor", "basket"]

OBJECTS = [f"{a} {n}" for a in ADJECTIVES for n in NOUNS]   # 38 x 35 = 1330


def nonsense_draws(n_draws, n_neg, seed=0):
    """n_draws DISTINCT groups of (1 positive + n_neg negatives) object phrases.

    Distinct positives matter: in 'ref' mode the negative is fixed, so the number
    of distinct positives is the number of distinct null values you can observe.
    """
    if n_draws > len(OBJECTS):
        raise ValueError(f"{n_draws} draws requested but only {len(OBJECTS)} "
                         f"distinct positives exist. Widen ADJECTIVES/NOUNS.")
    rng = random.Random(seed)
    positives = rng.sample(OBJECTS, n_draws)          # distinct, no repeats
    draws = []
    for pos in positives:
        negs = rng.sample([o for o in OBJECTS if o != pos], n_neg)
        draws.append(("a photo of a " + pos,
                      ["a photo of a " + o for o in negs]))
    return draws


def null_runs(img_emb, concepts, idx, split_ids, neg_mode, wording_table,
              ensemble=False, n_draws=1000, seed=0):
    """FIX 4 -- the null did not match the scoring machinery it was calibrating.

    Your null always built softmax(random_a vs random_b): one positive, one
    negative. But antonym_set mode divides by a SUM of two or three negatives,
    which changes the shape of the score. Calibrating a three-negative score
    against a one-negative null compares two different machines.

    Here the null is rebuilt per mode with the matching number of negatives.
    For 'ref' mode the real reference is kept in the negative slot and only the
    positive is replaced by nonsense -- that is the exact counterfactual you want:
    'what if the concept phrase carried no information?'
    """
    from sklearn.metrics import roc_auc_score
    mask = np.isin(idx, list(split_ids))
    y = {c: concepts.loc[idx[mask], c].to_numpy() for c in CRITERIA}
    runs = {c: [] for c in CRITERIA}

    for c in CRITERIA:
        n_neg = 1 if neg_mode in ("ref", "antonym") else len(ANTONYM_SET[c])
        for pos_txt, neg_txts in nonsense_draws(n_draws, n_neg, seed):
            pos_v = _vec(pos_txt, ensemble)
            if neg_mode == "ref":
                neg_vs = [_vec(REFERENCE, ensemble)]
            else:
                neg_vs = [_vec(t, ensemble) for t in neg_txts]
            s = _softmax_score(img_emb, pos_v, neg_vs)[mask]
            runs[c].append(roc_auc_score(y[c], s))
    return runs


# --------------------------------------------------------------------------
# EVALUATION
# --------------------------------------------------------------------------

def evaluate_vs_null(scores, concepts, idx, split_ids, runs, n_tests):
    """FIX 5 -- n_tests defaulted to 7 but you run far more than 7 tests.

    Bonferroni divides your 0.05 by the number of tests in the whole experiment,
    not the number in one table. 3 wordings x 3 neg_modes x 2 ensemble x 7
    concepts = 126. At n_tests=7 the threshold is 0.0071; the correct one is
    0.0004. With 200 null draws the smallest p you can even observe is 0.005,
    so at the correct threshold NOTHING can reach significance -- see the
    warning below. That is a real finding about your design, not a bug.
    """
    from sklearn.metrics import roc_auc_score
    mask = np.isin(idx, list(split_ids))
    alpha = 0.05 / n_tests
    rows = []
    for c in CRITERIA:
        y = concepts.loc[idx[mask], c].to_numpy()
        auc = roc_auc_score(y, scores[c][mask])
        null = np.asarray(runs[c])
        beaten = int((null >= auc).sum())
        # +1 / +1 is the standard permutation-test correction: an observed
        # statistic can never have a true p of exactly zero.
        p = (beaten + 1) / (len(null) + 1)
        rows.append({"concept": c, "n_pos": int(y.sum()), "auroc": round(auc, 3),
                     "null_med": round(float(np.median(null)), 3),
                     "null_p95": round(float(np.percentile(null, 95)), 3),
                     "beaten_by": beaten, "p": round(p, 4),
                     "sig": "YES" if p < alpha else ""})
    return pd.DataFrame(rows)


def alpha_report(n_tests, n_draws):
    lo = 1 / (n_draws + 1)
    alpha = 0.05 / n_tests
    print(f"tests={n_tests}  alpha={alpha:.2e}  smallest observable p={lo:.4f}")
    if lo >= alpha:
        need = int(np.ceil(1 / alpha)) - 1
        print(f"  WARNING: with {n_draws} null draws no result can clear this bar.")
        print(f"  Either raise n_draws to >= {need}, or pre-register a smaller grid,")
        print(f"  or correct within each table and say so explicitly in methods.")
    return alpha


# --------------------------------------------------------------------------
# SELFTEST -- runs with no CLIP, no data, no GPU
# --------------------------------------------------------------------------

def _selftest():
    print("SELFTEST: fake encoder, planted signal on 2 of 7 concepts.\n")
    rng = np.random.default_rng(0)
    n, d = 400, 64

    truth = pd.DataFrame(
        rng.binomial(1, 0.3, (n, len(CRITERIA))), columns=CRITERIA, index=range(n))
    idx = np.arange(n)
    split = set(range(n))

    # Two concept directions are genuinely encoded in the image vectors.
    dirs = rng.normal(size=(len(CRITERIA), d))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    img = rng.normal(size=(n, d))
    for j, c in enumerate(["blue_whitish_veil", "streaks"]):
        k = CRITERIA.index(c)
        img += 2.5 * truth[c].to_numpy()[:, None] * dirs[k][None, :]
    img /= np.linalg.norm(img, axis=1, keepdims=True)

    # Fake text encoder: a phrase containing a concept's keyword maps onto that
    # concept's direction; anything else maps to a random direction.
    KEY = {"veil": 1, "streak": 4}
    NEG_WORDS = ["no ", "typical", "symmetric", "uniform", "intact", "evenly"]

    def _is_negated(s):
        # careful: "irregular" CONTAINS "regular", and "atypical" contains
        # "typical". Substring matching on those flips the sign and you get a
        # perfectly inverted AUROC (~0.08 instead of ~0.89). Check the
        # affirmative forms first.
        s = s.lower()
        if "irregular" in s or "atypical" in s:
            return False
        return any(w in s for w in NEG_WORDS) or "regular" in s

    def fake(strings):
        out = []
        for s in strings:
            v = np.zeros(d)
            for kw, k in KEY.items():
                if kw in s.lower():
                    v = dirs[k] * (-1.0 if _is_negated(s) else 1.0)
            if not v.any():
                v = np.random.default_rng(abs(hash(s)) % 2**32).normal(size=d)
            out.append(v)
        return np.asarray(out)

    set_encoder(fake)
    wt = {"short": SHORT}["short"]

    alpha_report(n_tests=7 * 3 * 2, n_draws=1000)

    for mode in ("ref", "antonym", "antonym_set"):
        sc = concept_scores(img, wt, neg_mode=mode, ensemble=False)
        runs = null_runs(img, truth, idx, split, mode, wt, ensemble=False, n_draws=1000)
        res = evaluate_vs_null(sc, truth, idx, split, runs, n_tests=7 * 3 * 2)
        print(f"\n=== {mode} ===")
        print(res.to_string(index=False))
        hits = set(res.loc[res["auroc"] > 0.75, "concept"])
        print("recovered:", sorted(hits))
        assert {"blue_whitish_veil", "streaks"} <= hits, "planted signal not recovered"
        assert res["null_med"].between(0.4, 0.6).all(), "null is not centred on chance"
    print("\nselftest passed: planted concepts recovered, nulls centred on 0.5")


if __name__ == "__main__":
    _selftest()
