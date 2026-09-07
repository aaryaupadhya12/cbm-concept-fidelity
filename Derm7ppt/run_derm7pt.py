#!/usr/bin/env python3
"""
run_derm7pt.py -- the real run. No synthetic data anywhere in this file.

    python run_derm7pt.py --backbone clip   --stage all
    python run_derm7pt.py --backbone monet  --stage all --monet-ckpt /path/to/monet.pt
    python run_derm7pt.py --backbone clip   --stage checks     # data plumbing only, no GPU

STAGES
  checks   metadata, splits, positives vs published table, ceiling.  No GPU.
  encode   image embeddings -> cache.                                GPU.
  null     1000-draw random-text null on the eval split.             GPU (text only, fast).
  grid     the 6-cell short-label grid, corrected p-values.          GPU (text only).
  wording  short / taxonomy / clinical at the pre-registered cell.   GPU (text only).
  arms     four bottleneck arms + ceiling excess (H4).               No GPU.
  all      everything in order.

EVERY NUMBER IS COMPUTED ON THE TEST SPLIT (n~395) unless --eval-split says otherwise.
The valid split (n~203) puts the detection floor 0.024 higher for no reason, and
config selection already happened there.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import derm_audit_core as core
from derm_audit_core import CRITERIA, concept_scores, null_runs, evaluate_vs_null

# --------------------------------------------------------------------------
# PATHS -- edit these two
# --------------------------------------------------------------------------
DIR = Path("/kaggle/input/datasets/menakamohanakumar/derm7pt/release_v0")
WORK = Path("/kaggle/working")

# --------------------------------------------------------------------------
# BINARISATION -- clinical, from Argenziano. Never tuned to the data.
# These are the exact level strings in Derm7pt's meta.csv.
# --------------------------------------------------------------------------
POSITIVE = {
    "pigment_network":       {"atypical"},
    "blue_whitish_veil":     {"present"},
    "vascular_structures":   {"linear irregular", "dotted"},
    "pigmentation":          {"diffuse irregular", "localized irregular"},
    "streaks":               {"irregular"},
    "dots_and_globules":     {"irregular"},
    "regression_structures": {"blue areas", "white areas", "combinations"},
}

# Published counts over all 1011 images, from the consistency analysis.
# Used as a loader check: if these do not match, the binarisation drifted.
PUBLISHED_POSITIVES = {
    "dots_and_globules": 448, "pigmentation": 305, "regression_structures": 253,
    "streaks": 251, "pigment_network": 230, "blue_whitish_veil": 195,
    "vascular_structures": 71,
}
PUBLISHED_MULTIVALUED_CEILING = 0.921
PUBLISHED_MELANOMA = 252

# MONET is built on ORIGINAL OpenAI CLIP ViT-L/14, which uses QuickGELU.
# open_clip's plain "ViT-L-14" config sets quick_gelu=False and warns on the
# 'openai' tag -- that is the warning in your run log. Using it means your CLIP
# baseline runs a different activation than the weights were trained with AND a
# different one than MONET, so any CLIP-vs-MONET gap is partly an architecture
# mismatch. Use the quickgelu config for both.
MODELS = {
    "clip":        ("ViT-L-14-quickgelu", "openai"),   # correct baseline
    "clip_plain":  ("ViT-L-14", "openai"),             # the mismatched one, kept
                                                       # only to measure the gap
    "monet":       ("hf", "chanwkim/monet"),
}

# --------------------------------------------------------------------------
# DATA
# --------------------------------------------------------------------------

def load_meta(dir_release=DIR, verbose=True):
    meta_path = dir_release / "meta" / "meta.csv"
    if not meta_path.exists():
        sys.exit(f"meta.csv not found at {meta_path} -- is the dataset attached?")
    df = pd.read_csv(meta_path)
    if verbose:
        print(f"meta.csv: {len(df)} rows")

    # train/valid/test_indexes.csv hold POSITIONAL row numbers into the original
    # meta.csv. read_csv gives a RangeIndex 0..1010 so df.index matches them. If
    # anyone inserts a reset_index(), sort, or merge that renumbers rows, every
    # split silently points at the wrong images and nothing raises. Assert it.
    assert isinstance(df.index, pd.RangeIndex) and df.index[0] == 0, (
        "meta.csv index is not the original 0-based RangeIndex; split indexes "
        "will not line up. Do not reset_index() or reorder before this point.")
    if len(df) != 1011:
        print(f"  WARN expected 1011 rows, got {len(df)}")

    splits = {}
    for name in ("train", "valid", "test"):
        p = dir_release / "meta" / f"{name}_indexes.csv"
        if not p.exists():
            sys.exit(f"missing {p}")
        s = pd.read_csv(p)
        splits[name] = set(s[s.columns[0]].tolist())

    # Drop rows where clinic and derm point at the same file. We encode 'derm'
    # only, so after this it is one image per lesion and image-level folds are
    # correct -- no grouped CV needed.
    dup = df["clinic"].astype(str) == df["derm"].astype(str)
    if verbose:
        print(f"dropping {int(dup.sum())} flattened-modality rows -> "
              f"n={len(df) - int(dup.sum())}")
        if dup.any():
            # Your own sanity checker treats clinic==derm as a FAIL while the
            # loader silently drops it. Look at them once: if these cases simply
            # lack a dermoscopic image and the clinical photo was substituted,
            # dropping is right and belongs in methods as one sentence.
            cols = [c for c in ("case_num", "case_id", "diagnosis", "clinic") if c in df.columns]
            print("  the dropped rows:")
            print(df.loc[dup, cols].to_string(index=True))
    df = df.loc[~dup].copy()

    unmapped = {}
    cols = {}
    for c in CRITERIA:
        raw = df[c].astype(str).str.strip().str.lower()
        want = {v.lower() for v in POSITIVE[c]}
        seen = set(raw.unique())
        missing = want - seen
        if missing:
            unmapped[c] = {"expected_but_absent": sorted(missing),
                           "levels_present": sorted(seen)}
        cols[c] = raw.isin(want).astype(int)
    if unmapped:
        print("\n!! POSITIVE levels not found in the data:")
        print(json.dumps(unmapped, indent=2))
        sys.exit("binarisation does not match the level vocabulary. Fix POSITIVE.")

    concepts = pd.DataFrame(cols, index=df.index)
    mel = df["diagnosis"].astype(str).str.lower().str.contains("melanoma").astype(int)
    return df, concepts, mel, splits


def hard_ceiling(concepts, mel):
    g = concepts.assign(mel=mel).groupby(CRITERIA)["mel"].agg(["size", "sum"])
    correct = g.apply(lambda r: max(r["sum"], r["size"] - r["sum"]), axis=1).sum()
    return {"n": int(len(mel)), "profiles": int(len(g)),
            "inconsistent": int(((g["sum"] > 0) & (g["sum"] < g["size"])).sum()),
            "ceiling": round(float(correct / len(mel)), 4),
            "majority_baseline": round(float(max(mel.mean(), 1 - mel.mean())), 4)}


def stage_checks(df, concepts, mel, splits):
    print("\n=== DATA CHECKS ===")
    ok = True

    print("\npositives vs published (all rows kept in the published count, we drop 9):")
    for c in CRITERIA:
        got, pub = int(concepts[c].sum()), PUBLISHED_POSITIVES[c]
        flag = "ok" if abs(got - pub) <= 12 else "<-- MISMATCH"
        if flag != "ok":
            ok = False
        print(f"  {c:24s} ours={got:4d}  published={pub:4d}  {flag}")

    print(f"\nmelanoma: {int(mel.sum())} (published {PUBLISHED_MELANOMA}), "
          f"prevalence {mel.mean():.4f}")
    print(f"\ndiagnosis: {df['diagnosis'].nunique()} raw classes")
    print("  melanoma=1:")
    for d in sorted(df.loc[mel == 1, "diagnosis"].astype(str).unique()):
        print(f"    {d:38s} n={int((df['diagnosis'] == d).sum())}")
    print("  melanoma=0:")
    for d in sorted(df.loc[mel == 0, "diagnosis"].astype(str).unique()):
        print(f"    {d:38s} n={int((df['diagnosis'] == d).sum())}")
    print("  CHECK: 'melanosis' must appear under melanoma=0. It does not contain")
    print("  the substring 'melanoma', so contains() is safe -- but confirm by eye.")

    img_dir = DIR / "images"
    sample = df["derm"].astype(str).head(25)
    hits = sum((img_dir / r).exists() for r in sample)
    if hits == len(sample):
        print(f"\nimage paths: sampled {len(sample)}/{len(sample)} resolve under images/")
    else:
        print(f"\n  !! only {hits}/{len(sample)} derm paths resolve under {img_dir}")
        ok = False

    print(f"one image per lesion: rows={len(df)} unique derm paths={df['derm'].nunique()}")
    if len(df) != df["derm"].nunique():
        print("  !! a lesion appears twice -- grouped CV IS required"); ok = False

    for a, b in (("train", "valid"), ("train", "test"), ("valid", "test")):
        if splits[a] & splits[b]:
            print(f"  !! {a}/{b} splits overlap -- leakage"); ok = False
    for name, ids in splits.items():
        print(f"  {name}: {len(df.index.isin(list(ids)).nonzero()[0])} of our rows")

    ceil = hard_ceiling(concepts, mel)
    print("\nceiling:", json.dumps(ceil, indent=2))
    if ceil["ceiling"] > PUBLISHED_MULTIVALUED_CEILING:
        print("  !! ABOVE the 92.1% full-multivalued ceiling. Impossible: collapsing")
        print("     levels merges profiles and can only LOWER the bound. Stop and fix.")
        ok = False
    else:
        print(f"  ok, below the published {PUBLISHED_MULTIVALUED_CEILING} as required")

    print("\npositives in each split (concepts with <10 are reported, never selected on):")
    for name, ids in splits.items():
        m = df.index.isin(list(ids))
        counts = concepts.loc[m].sum()
        thin = [c for c in CRITERIA if counts[c] < 10]
        print(f"  {name:5s} " + "  ".join(f"{c[:12]}={int(counts[c])}" for c in CRITERIA))
        if thin:
            print(f"        thin: {thin}")

    print("\n" + ("ALL CHECKS PASSED" if ok else "CHECKS FAILED -- do not run experiments"))
    return ok, ceil


# --------------------------------------------------------------------------
# ENCODERS -- the only part not verifiable without the real weights
# --------------------------------------------------------------------------

def _sanity_check(bb, tag):
    """Catch a broken backbone BEFORE spending 15 minutes encoding 1002 images.

    Three failures this catches, all of which otherwise produce plausible-looking
    numbers rather than an exception:
      1. image and text embeddings in different spaces (wrong projection used)
      2. embeddings not unit-normalised
      3. a text encoder that returns the same vector for every string
    """
    import numpy as _np
    from PIL import Image as _Im
    probe = [_Im.new("RGB", (224, 224), c) for c in ("red", "blue")]
    vi = bb.encode_images(probe)
    vt = bb.encode_texts(["a photo of a red square", "a photo of a blue square",
                          "a dermoscopic image of a skin lesion"])

    assert vi.shape[1] == vt.shape[1], (
        f"{tag}: image dim {vi.shape[1]} != text dim {vt.shape[1]}. The wrong "
        f"projection was applied -- similarities would be meaningless.")
    for nm, v in (("image", vi), ("text", vt)):
        n = _np.linalg.norm(v, axis=1)
        assert _np.allclose(n, 1.0, atol=1e-3), f"{tag}: {nm} vectors not unit norm ({n})"
    spread = float(_np.abs(vt @ vt.T - _np.eye(len(vt))).max())
    assert spread < 0.999, f"{tag}: text encoder returns near-identical vectors"

    sims = vi @ vt.T
    print(f"  sanity: dim={vi.shape[1]}, cos range [{sims.min():.3f}, {sims.max():.3f}], "
          f"red-vs-blue text separation {sims[0,0]-sims[0,1]:+.3f}")
    if sims[0, 0] <= sims[0, 1]:
        print("  WARN: red image is not closer to 'red square' than to 'blue square'.")
        print("        The backbone may be misloaded. Investigate before trusting scores.")


MONET_WEIGHTS_URL = "https://aimslab.cs.washington.edu/MONET/weight_clip.pt"


def monet_transform(n_px=224):
    """MONET's own preprocessing, copied from their README.

    NOTE the normalisation: ImageNet statistics (0.485/0.456/0.406,
    0.229/0.224/0.225), NOT CLIP's (0.481/0.458/0.408, 0.269/0.261/0.276).
    Using CLIP's numbers here would silently degrade every MONET score and the
    comparison would be measuring preprocessing, not pretraining.
    """
    import torchvision.transforms as T
    return T.Compose([
        T.Resize(n_px, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(n_px),
        lambda im: im.convert("RGB"),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])


def _build_monet(tag, dev):
    """MONET, via the route its authors document.

    Order of attempts:
      1. openai/CLIP package + weight_clip.pt        <- what the MONET README says
      2. open_clip ViT-L-14-quickgelu + same weights <- same architecture, no extra install
      3. HuggingFace chanwkim/monet                  <- last resort; on transformers
                                                        4.5x the auto-class can build a
                                                        vision tower whose hidden size
                                                        (768) does not match the
                                                        projection head (1024x768)

    MONET is ViT-L/14 from the original OpenAI implementation, so QuickGELU
    throughout. Do not load these weights into a non-quickgelu config.
    """
    import torch

    try:
        import clip as openai_clip
        model, _ = openai_clip.load("ViT-L/14", device=dev, jit=False)
        sd = torch.hub.load_state_dict_from_url(MONET_WEIGHTS_URL, map_location="cpu")
        sd = sd.get("state_dict", sd)
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"MONET via openai/CLIP: {len(missing)} missing, {len(unexpected)} unexpected")
        if len(missing) > 20:
            raise RuntimeError(f"{len(missing)} missing keys -- refusing to trust this load")
        model = model.eval().to(dev)
        pre = monet_transform(224)

        def encode_images(pil_batch):
            with torch.no_grad():
                x = torch.stack([pre(im) for im in pil_batch]).to(dev)
                f = model.encode_image(x)
                return (f / f.norm(dim=-1, keepdim=True)).float().cpu().numpy()

        def encode_texts(strings):
            with torch.no_grad():
                t = model.encode_text(openai_clip.tokenize(list(strings), truncate=True).to(dev))
                return (t / t.norm(dim=-1, keepdim=True)).float().cpu().numpy()

        bb = Backbone(encode_images, encode_texts, "monet")
        _sanity_check(bb, "monet/openai-clip")
        return bb
    except ImportError:
        print("openai/CLIP not installed. For the authors' exact path run:")
        print("  !pip install -q --no-deps git+https://github.com/openai/CLIP.git")
        print("Falling back to open_clip with the same weights.")
    except Exception as e:
        print(f"openai/CLIP route failed: {e}. Falling back to open_clip.")

    try:
        import open_clip
        ckpt = torch.hub.load_state_dict_from_url(MONET_WEIGHTS_URL, map_location="cpu")
        ckpt = ckpt.get("state_dict", ckpt)
        ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
        model, _, _ = open_clip.create_model_and_transforms("ViT-L-14-quickgelu")
        missing, unexpected = model.load_state_dict(ckpt, strict=False)
        print(f"MONET via open_clip: {len(missing)} missing, {len(unexpected)} unexpected")
        if len(missing) > 20:
            raise RuntimeError(f"{len(missing)} missing keys -- weights are not landing")
        model = model.eval().to(dev)
        tok = open_clip.get_tokenizer("ViT-L-14-quickgelu")
        pre = monet_transform(224)

        def encode_images(pil_batch):
            with torch.no_grad():
                x = torch.stack([pre(im) for im in pil_batch]).to(dev)
                f = model.encode_image(x)
                return (f / f.norm(dim=-1, keepdim=True)).float().cpu().numpy()

        def encode_texts(strings):
            with torch.no_grad():
                t = model.encode_text(tok(list(strings)).to(dev))
                return (t / t.norm(dim=-1, keepdim=True)).float().cpu().numpy()

        bb = Backbone(encode_images, encode_texts, "monet")
        _sanity_check(bb, "monet/open_clip-quickgelu")
        return bb
    except Exception as e:
        print(f"open_clip route failed: {e}. Trying HuggingFace.")

    from transformers import AutoProcessor, CLIPModel
    proc = AutoProcessor.from_pretrained(tag)
    model = CLIPModel.from_pretrained(tag).to(dev).eval()
    vc, tc = model.config.vision_config, model.config.text_config
    print(f"HF config: vision hidden={vc.hidden_size}, text hidden={tc.hidden_size}, "
          f"projection={model.config.projection_dim}")
    if vc.hidden_size != 1024:
        raise RuntimeError(
            f"vision hidden size {vc.hidden_size} != 1024. MONET is ViT-L/14, so this "
            f"repo did not build the right architecture on your transformers version. "
            f"Use the openai/CLIP route instead:\n"
            f"  !pip install -q --no-deps git+https://github.com/openai/CLIP.git")

    def encode_images(pil_batch):
        with torch.no_grad():
            x = proc(images=pil_batch, return_tensors="pt").to(dev)
            f = model.get_image_features(**x)
            return (f / f.norm(dim=-1, keepdim=True)).float().cpu().numpy()

    def encode_texts(strings):
        with torch.no_grad():
            x = proc(text=list(strings), return_tensors="pt", padding=True,
                     truncation=True).to(dev)
            t = model.get_text_features(**x)
            return (t / t.norm(dim=-1, keepdim=True)).float().cpu().numpy()

    bb = Backbone(encode_images, encode_texts, "monet")
    _sanity_check(bb, "monet/hf")
    return bb


class Backbone:
    """Uniform interface over open_clip and HuggingFace CLIP."""

    def __init__(self, encode_images, encode_texts, name):
        self.encode_images = encode_images
        self.encode_texts = encode_texts
        self.name = name


def build_encoder(backbone, monet_ckpt=None):
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    arch, tag = MODELS[backbone]

    if arch == "hf":
        return _build_monet(tag, dev)

    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained=tag)
    model = model.to(dev).eval()
    tok = open_clip.get_tokenizer(arch)
    print(f"loaded open_clip {arch} / {tag}")

    def encode_images(pil_batch):
        with torch.no_grad():
            x = torch.stack([preprocess(im) for im in pil_batch]).to(dev)
            f = model.encode_image(x)
            return (f / f.norm(dim=-1, keepdim=True)).float().cpu().numpy()

    def encode_texts(strings):
        with torch.no_grad():
            t = model.encode_text(tok(list(strings)).to(dev))
            return (t / t.norm(dim=-1, keepdim=True)).float().cpu().numpy()

    bb = Backbone(encode_images, encode_texts, backbone)
    _sanity_check(bb, f"{arch}/{tag}")
    return bb


def stage_encode(df, backbone, monet_ckpt=None, batch=32, force=False):
    from PIL import Image
    cache = WORK / "cache"; cache.mkdir(parents=True, exist_ok=True)
    emb_p, idx_p = cache / f"img_emb_{backbone}.npy", cache / f"img_idx_{backbone}.npy"

    bb = build_encoder(backbone, monet_ckpt)
    core.set_encoder(bb.encode_texts)
    core.clear_vec_cache()

    if emb_p.exists() and not force:
        print(f"using cached {emb_p.name}")
        return np.load(emb_p), np.load(idx_p)

    rels = df["derm"].astype(str).tolist()
    out = []
    for i in range(0, len(rels), batch):
        ims = [Image.open(DIR / "images" / r).convert("RGB") for r in rels[i:i + batch]]
        out.append(bb.encode_images(ims))
        print(f"  {min(i + batch, len(rels))}/{len(rels)}", end="\r")
    emb, idx = np.concatenate(out), np.asarray(df.index)
    np.save(emb_p, emb); np.save(idx_p, idx)
    print(f"\ncached {emb.shape} -> {emb_p}")
    return emb, idx


CELLS = [(m, e) for m in ("ref", "antonym", "antonym_set") for e in (False, True)]
PREREG_CELL = ("antonym", True)


def stage_grid(img_emb, concepts, idx, eval_ids, backbone, n_draws=1000):
    n_tests = len(CELLS) * len(CRITERIA)
    core.alpha_report(n_tests=n_tests, n_draws=n_draws)

    nulls, tables = {}, []
    for mode, ens in CELLS:
        key = (mode, ens)
        if key not in nulls:
            print(f"\nnull for {mode} ens={ens} ({n_draws} draws)...", flush=True)
            nulls[key] = null_runs(img_emb, concepts, idx, eval_ids, mode,
                                   core.SHORT, ensemble=ens, n_draws=n_draws)
        sc = concept_scores(img_emb, core.SHORT, neg_mode=mode, ensemble=ens)
        r = evaluate_vs_null(sc, concepts, idx, eval_ids, nulls[key], n_tests=n_tests)
        r.insert(0, "cell", f"{mode}_{'ens' if ens else 'noens'}")
        r["floor_p95"] = [np.percentile(nulls[key][c], 95).round(3) for c in CRITERIA]
        r["headroom"] = (r["auroc"] - r["floor_p95"]).round(3)
        tables.append(r)
        print(f"\n=== {backbone} :: {r['cell'].iloc[0]} ===")
        print(r.to_string(index=False))

    grid = pd.concat(tables, ignore_index=True)
    grid["backbone"] = backbone
    grid.to_csv(WORK / f"h1_grid_{backbone}.csv", index=False)

    sel = grid[grid.concept != "vascular_structures"]
    print(f"\nmean AUROC per cell (vascular excluded):")
    print(sel.groupby("cell")["auroc"].mean().sort_values(ascending=False).round(4).to_string())
    print(f"\nconcepts clearing the null anywhere: {int((grid['sig'] == 'YES').sum())}")
    return grid, nulls


def stage_wording(img_emb, concepts, idx, eval_ids, nulls, backbone,
                  clinical_json="clinical_prompts.json"):
    mode, ens = PREREG_CELL
    wordings = core.load_wordings(clinical_json)
    if "clinical" not in wordings:
        sys.exit(f"{clinical_json} not found or missing tiers")

    from sklearn.metrics import roc_auc_score
    mask = np.isin(idx, list(eval_ids))
    rows = []
    for tier, table in wordings.items():
        sc = concept_scores(img_emb, table, neg_mode=mode, ensemble=ens)
        for c in CRITERIA:
            y = concepts.loc[idx[mask], c].to_numpy()
            rows.append({"backbone": backbone, "tier": tier, "concept": c,
                         "auroc": round(roc_auc_score(y, sc[c][mask]), 3)})
    long = pd.DataFrame(rows)
    wide = long.pivot(index="concept", columns="tier", values="auroc")
    wide = wide[[t for t in ("short", "taxonomy", "clinical") if t in wide.columns]]
    wide["spread"] = (wide.max(axis=1) - wide.min(axis=1)).round(3)
    wide["best"] = wide[[c for c in wide.columns if c != "spread"]].idxmax(axis=1)

    print(f"\n=== WORDING ABLATION ({backbone}, cell {mode}+{'ens' if ens else 'noens'}) ===")
    print(wide.to_string())
    sel = wide.drop(index="vascular_structures", errors="ignore")
    print(f"\nmean wording spread (vascular excluded): {sel['spread'].mean():.3f}")
    if "pigmentation" in wide.index:
        r = wide.loc["pigmentation"]
        print(f"pigmentation (pre-specified): short={r.get('short')} "
              f"clinical={r.get('clinical')}")
    long.to_csv(WORK / f"wording_{backbone}.csv", index=False)
    return wide


def stage_arms(img_emb, concepts, mel, idx, splits, ceil, backbone):
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    mode, ens = PREREG_CELL
    sc = concept_scores(img_emb, core.SHORT, neg_mode=mode, ensemble=ens)
    V = np.column_stack([sc[c] for c in CRITERIA])
    E = concepts.loc[idx, CRITERIA].to_numpy()
    y = mel.loc[idx].to_numpy()

    dev = np.isin(idx, list(splits["train"] | splits["valid"]))
    te = np.isin(idx, list(splits["test"]))
    rng = np.random.default_rng(0)

    def binarise(Vtr, Vte, Etr):
        Btr, Bte = np.zeros_like(Vtr, int), np.zeros_like(Vte, int)
        for j in range(Vtr.shape[1]):
            thr = np.quantile(Vtr[:, j], 1 - Etr[:, j].mean())
            Btr[:, j], Bte[:, j] = Vtr[:, j] >= thr, Vte[:, j] >= thr
        return Btr.astype(float), Bte.astype(float)

    arms = {}
    arms["expert"] = (E[dev].astype(float), E[te].astype(float))
    arms["vlm"] = binarise(V[dev], V[te], E[dev])
    # One permutation is a sample, not a control. Build a distribution.
    N_SHUFFLE = 50   # 10 gives a p-value floor of 0.091, which can never clear 0.05
    shuffle_arms = []
    for s_ in range(N_SHUFFLE):
        r = np.random.default_rng(1000 + s_)
        Vs_d = np.column_stack([r.permutation(V[dev][:, j]) for j in range(V.shape[1])])
        Vs_t = np.column_stack([r.permutation(V[te][:, j]) for j in range(V.shape[1])])
        shuffle_arms.append(binarise(Vs_d, Vs_t, E[dev]))
    arms["vlm_shuffled"] = shuffle_arms[0]
    mu, sd = V[dev].mean(0), V[dev].std(0) + 1e-8
    arms["vlm_continuous"] = ((V[dev] - mu) / sd, (V[te] - mu) / sd)

    print(f"\n=== FOUR ARMS ({backbone}, MLP head, test split) ===")
    print(f"hard-concept ceiling = {ceil['ceiling']:.4f}   "
          f"majority baseline = {ceil['majority_baseline']:.4f}\n")
    res = {}
    for name, (Xtr, Xte) in arms.items():
        m = make_pipeline(StandardScaler(),
                          MLPClassifier((32,), max_iter=3000, alpha=1e-3, random_state=0))
        m.fit(Xtr, y[dev]); p = m.predict(Xte)
        acc = accuracy_score(y[te], p); f1 = f1_score(y[te], p, average="macro")
        excess = acc - ceil["ceiling"]
        res[name] = f1
        if name == "vlm_continuous":
            acc_cont = acc
        if name == "vlm_continuous":
            flag = "  <-- exceeds ceiling: H4 LEAKAGE EVIDENCE" if excess > 0 else ""
        else:
            flag = "  <-- ABOVE CEILING ON HARD CONCEPTS: bug" if excess > 0 else ""
        print(f"  {name:16s} acc={acc:.4f}  macroF1={f1:.4f}  excess={excess:+.4f}{flag}")

    shuffled_null = []
    for Xtr, Xte in shuffle_arms:
        m = make_pipeline(StandardScaler(),
                          MLPClassifier((32,), max_iter=3000, alpha=1e-3, random_state=0))
        m.fit(Xtr, y[dev])
        shuffled_null.append(f1_score(y[te], m.predict(Xte), average="macro"))
    print(f"\n  shuffled control over {len(shuffled_null)} permutations: "
          f"macroF1 {np.mean(shuffled_null):.4f} +/- {np.std(shuffled_null):.4f}")

    always_majority = 2 * (1 - y[te].mean()) / (2 - y[te].mean()) / 2
    print(f"  always-majority classifier macroF1 = {always_majority:.4f} "
          f"(any arm near this has learned nothing)")

    pd.DataFrame([{"backbone": backbone, "arm": k, "macro_f1": v,
                   "ceiling": ceil["ceiling"], "majority": ceil["majority_baseline"]}
                  for k, v in res.items()]
                 ).to_csv(WORK / f"arms_{backbone}.csv", index=False)
    pd.Series(shuffled_null).to_csv(WORK / f"arms_{backbone}_shufflenull.csv", index=False)
    print(f"  wrote arms_{backbone}.csv")

    from protocol import leakage_verdict
    res["vlm_continuous_acc"] = acc_cont
    print("\n" + leakage_verdict(res, ceil["ceiling"], shuffled_null=shuffled_null))
    return res


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="clip", choices=list(MODELS))
    ap.add_argument("--stage", default="all",
                    choices=["checks", "encode", "grid", "wording", "arms", "all"])
    ap.add_argument("--eval-split", default="test", choices=["valid", "test"])
    ap.add_argument("--n-draws", type=int, default=1000)
    ap.add_argument("--monet-ckpt", default=None)
    ap.add_argument("--dir", type=Path, default=None)
    ap.add_argument("--clinical-json", default="clinical_prompts.json")
    a = ap.parse_args()

    global DIR
    if a.dir is not None:
        DIR = a.dir
    WORK.mkdir(parents=True, exist_ok=True)

    df, concepts, mel, splits = load_meta(DIR)
    ok, ceil = stage_checks(df, concepts, mel, splits)
    if not ok and a.stage != "checks":
        sys.exit("checks failed; refusing to run experiments")
    if a.stage == "checks":
        return

    img_emb, idx = stage_encode(df, a.backbone, a.monet_ckpt)

    eval_ids = splits[a.eval_split]
    print(f"\nevaluating on the {a.eval_split} split "
          f"({df.index.isin(list(eval_ids)).sum()} of our rows)")

    nulls = None
    if a.stage in ("grid", "all"):
        _, nulls = stage_grid(img_emb, concepts, idx, eval_ids, a.backbone, a.n_draws)
    if a.stage in ("wording", "all"):
        stage_wording(img_emb, concepts, idx, eval_ids, nulls, a.backbone, a.clinical_json)
    if a.stage in ("arms", "all"):
        stage_arms(img_emb, concepts, mel, idx, splits, ceil, a.backbone)


def run_all(backbone="clip", stage="all", eval_split="test", n_draws=1000,
            monet_ckpt=None, dir_release=None, clinical_json="clinical_prompts.json"):
    """Notebook entry point. Call from a Kaggle cell instead of the CLI:

        import run_derm7pt as R
        df, concepts, mel, splits, ceil = R.run_all(stage="checks")
    """
    global DIR
    if dir_release is not None:
        DIR = Path(dir_release)
    WORK.mkdir(parents=True, exist_ok=True)

    df, concepts, mel, splits = load_meta(DIR)
    ok, ceil = stage_checks(df, concepts, mel, splits)
    if not ok and stage != "checks":
        raise RuntimeError("checks failed; refusing to run experiments")
    if stage == "checks":
        return df, concepts, mel, splits, ceil

    img_emb, idx = stage_encode(df, backbone, monet_ckpt)

    eval_ids = splits[eval_split]
    nulls = None
    if stage in ("grid", "all"):
        _, nulls = stage_grid(img_emb, concepts, idx, eval_ids, backbone, n_draws)
    if stage in ("wording", "all"):
        stage_wording(img_emb, concepts, idx, eval_ids, nulls, backbone, clinical_json)
    if stage in ("arms", "all"):
        stage_arms(img_emb, concepts, mel, idx, splits, ceil, backbone)
    return df, concepts, mel, splits, ceil


if __name__ == "__main__":
    main()
