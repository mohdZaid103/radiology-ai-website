#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Radiology AI Assistant -- Explainable Chest X-Ray Classification
================================================================
CSET400C Capstone (Milestone 1 report implementation)

Classes   : Normal | COVID-19 | Pneumonia | Tuberculosis
Pipeline  : multi-source dataset audit -> leakage-safe split -> contrast
            experiments (CLAHE / augmentation) -> Model A (DenseNet121 transfer
            learning head) -> Model B (deep-feature + SVC) -> evaluation ->
            Grad-CAM (CNN branch only) -> FastAPI inference service.

NOT A MEDICAL DEVICE.  Research / teaching prototype.  Outputs are dataset-
defined labels and must never be used for diagnosis, triage or treatment.

Datasets (download manually from Kaggle / NIH, check each licence first)
-----------------------------------------------------------------------
 1. covid_radiography : COVID-19 Radiography Database (Rahman et al.)
      kaggle: tawsifurrahman/covid19-radiography-database
      uses COVID, Normal, Viral Pneumonia.  "Lung_Opacity" is EXCLUDED on
      purpose (the report says it must not be equated with pneumonia).
 2. kermany_pneumonia : Chest X-Ray Images (Pneumonia)  (Kermany et al.)
      kaggle: paultimothymooney/chest-xray-pneumonia
      NORMAL + PNEUMONIA (bacterial + viral -> broader than source 1's
      "Viral Pneumonia"; this is recorded as a limitation).
 3. tb_kaggle : Tuberculosis (TB) Chest X-ray Database (Rahman et al.)
      kaggle: tawsifurrahman/tuberculosis-tb-chest-xray-dataset
 4. shenzhen : Shenzhen Hospital TB set (NLM)  - optional
      files like CHNCXR_0001_0.png   (_0 = normal, _1 = TB)
 5. montgomery : Montgomery County TB set (NLM) - optional
      files like MCUCXR_0001_0.png   (_0 = normal, _1 = TB)

Expected layout (override any folder with --source name=path)
    <data-root>/COVID-19_Radiography_Dataset/{COVID,Normal,Viral Pneumonia,...}
    <data-root>/chest_xray/{train,val,test}/{NORMAL,PNEUMONIA}
    <data-root>/TB_Chest_Radiography_Database/{Normal,Tuberculosis}
    <data-root>/shenzhen/**/CHNCXR_*.png
    <data-root>/montgomery/**/MCUCXR_*.png

Install
-------
    pip install torch torchvision opencv-python-headless scikit-learn pandas \
                numpy matplotlib joblib fastapi uvicorn python-multipart

Typical usage
-------------
    python radiology_ai_assistant.py prepare   --data-root ./data --max-per-class 4000
    python radiology_ai_assistant.py clahe-demo
    python radiology_ai_assistant.py run-all                  # all presets, CNN + SVC
    # ...or step by step:
    python radiology_ai_assistant.py train-cnn --preset clahe_aug
    python radiology_ai_assistant.py train-svc --preset clahe_aug
    python radiology_ai_assistant.py evaluate  --model cnn --preset clahe_aug
    python radiology_ai_assistant.py evaluate  --model svc --preset clahe_aug
    python radiology_ai_assistant.py gradcam   --preset clahe_aug --image some_xray.png
    python radiology_ai_assistant.py serve     --preset clahe_aug --port 8000
"""
from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
import logging
import math
import platform
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVC  # noqa: E402
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler  # noqa: E402
from torchvision import models  # noqa: E402

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
MODEL_VERSION = "0.1.0"
CLASSES = ["Normal", "COVID-19", "Pneumonia", "Tuberculosis"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
DISCLAIMER = (
    "Research prototype - NOT for clinical use. Output is a dataset-defined "
    "label from a student model, not a diagnosis, triage decision or "
    "treatment recommendation. A qualified clinician must review any image."
)
GRADCAM_NOTE = (
    "The heatmap shows image regions that raised the CNN score for the "
    "predicted class. It is a coarse visual aid, not a lesion segmentation "
    "and not proof that the model reasons correctly."
)

log = logging.getLogger("radiology_ai")


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def env_versions() -> Dict[str, str]:
    import sklearn

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "opencv": cv2.__version__,
        "sklearn": sklearn.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))


def decode_gray(buf: np.ndarray) -> Optional[np.ndarray]:
    """Decode bytes -> uint8 grayscale (handles 16-bit PNG/TIFF). None if unreadable."""
    img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 3:
        img = cv2.cvtColor(img[..., :3], cv2.COLOR_BGR2GRAY) if img.shape[2] >= 3 else img[..., 0]
    if img.dtype != np.uint8:
        img = img.astype(np.float32)
        lo, hi = float(img.min()), float(img.max())
        img = ((img - lo) / max(hi - lo, 1e-6) * 255.0).astype(np.uint8)
    return img


def read_gray(path) -> Optional[np.ndarray]:
    try:
        return decode_gray(np.fromfile(str(path), dtype=np.uint8))
    except Exception:  # unreadable / truncated file
        return None


# --------------------------------------------------------------------------- #
# Dataset sources & label mapping  (report sec. 7.2)
# --------------------------------------------------------------------------- #
def _by_folder(mapping: Dict[str, Optional[str]]) -> Callable[[Path], Optional[str]]:
    """Label from the closest parent folder found in `mapping` (case-insensitive).
    A mapping value of None means 'explicitly excluded'."""

    def fn(rel: Path) -> Optional[str]:
        for part in reversed([p.lower() for p in rel.parts[:-1]]):
            if part in mapping:
                return mapping[part]
        return None

    return fn


def _by_suffix(rel: Path) -> Optional[str]:
    """Shenzhen / Montgomery convention: ..._0 normal, ..._1 TB."""
    if any("mask" in p.lower() for p in rel.parts):
        return None
    m = re.search(r"_([01])$", rel.stem)
    if not m:
        return None
    return "Normal" if m.group(1) == "0" else "Tuberculosis"


def _stem_group(rel: Path) -> str:
    return rel.stem


def _kermany_group(rel: Path) -> str:
    s = rel.stem
    m = re.match(r"(person\d+)", s, flags=re.I) or re.search(r"(IM-\d+)", s)
    return m.group(1) if m else s


def _suffix_group(rel: Path) -> str:
    return re.sub(r"_[01]$", "", rel.stem)


@dataclass
class Source:
    name: str
    subdir: str
    label_fn: Callable[[Path], Optional[str]]
    group_fn: Callable[[Path], str]
    note: str = ""


SOURCES: Dict[str, Source] = {
    "covid_radiography": Source(
        "covid_radiography",
        "COVID-19_Radiography_Dataset",
        _by_folder({"covid": "COVID-19", "normal": "Normal", "viral pneumonia": "Pneumonia",
                    "lung_opacity": None, "masks": None}),
        _stem_group,
        "Lung_Opacity excluded; no patient IDs -> de-duplication by hash only.",
    ),
    "kermany_pneumonia": Source(
        "kermany_pneumonia",
        "chest_xray",
        _by_folder({"normal": "Normal", "pneumonia": "Pneumonia", "__macosx": None}),
        _kermany_group,
        "PNEUMONIA = bacterial + viral (broader than 'Viral Pneumonia'). Original split ignored.",
    ),
    "tb_kaggle": Source(
        "tb_kaggle",
        "TB_Chest_Radiography_Database",
        _by_folder({"normal": "Normal", "tuberculosis": "Tuberculosis"}),
        _stem_group,
        "Normals partly re-used from other public sets -> hash de-dup is essential.",
    ),
    "shenzhen": Source("shenzhen", "shenzhen", _by_suffix, _suffix_group, "Optional NLM TB set."),
    "montgomery": Source("montgomery", "montgomery", _by_suffix, _suffix_group, "Optional NLM TB set."),
}


def _file_fingerprint(path: Path) -> Tuple[str, str, int, int]:
    """md5 of file bytes + 64-bit difference hash (robust to re-encoding/resizing)."""
    raw = np.fromfile(str(path), dtype=np.uint8)
    md5 = hashlib.md5(raw.tobytes()).hexdigest()
    img = decode_gray(raw)
    if img is None:
        return md5, "", 0, 0
    h, w = img.shape
    small = cv2.resize(img, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    dhash = "".join("1" if b else "0" for b in bits)
    return md5, f"{int(dhash, 2):016x}", w, h


def build_manifest(data_root: Path, overrides: Dict[str, str], only: Optional[List[str]],
                   max_per_class: Optional[int], seed: int, workers: int = 8) -> pd.DataFrame:
    rows = []
    for name, src in SOURCES.items():
        if only and name not in only:
            continue
        root = Path(overrides.get(name, data_root / src.subdir))
        if not root.exists():
            log.warning("source %-18s skipped (folder not found: %s)", name, root)
            continue
        n0 = len(rows)
        for p in sorted(root.rglob("*")):
            if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
                continue
            rel = p.relative_to(root)
            label = src.label_fn(rel)
            if label is None:
                continue
            rows.append({"path": str(p), "source": name, "label": label,
                         "group": f"{name}:{src.group_fn(rel)}"})
        log.info("source %-18s %6d labelled images  (%s)", name, len(rows) - n0, src.note)
    if not rows:
        raise SystemExit("No images found. Check --data-root / --source arguments.")
    df = pd.DataFrame(rows)

    log.info("fingerprinting %d images (md5 + dHash) ...", len(df))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        fps = list(ex.map(lambda s: _file_fingerprint(Path(s)), df["path"]))
    df["md5"], df["dhash"], df["width"], df["height"] = zip(*fps)
    bad = df["dhash"] == ""
    if bad.any():
        log.warning("dropping %d unreadable images", int(bad.sum()))
        df = df[~bad]

    # Conflicting labels for visually identical images -> drop all copies.
    conflict = df.groupby("dhash")["label"].transform("nunique") > 1
    if conflict.any():
        log.warning("dropping %d images whose duplicates carry conflicting labels", int(conflict.sum()))
        df = df[~conflict]
    before = len(df)
    df = df.drop_duplicates("dhash", keep="first")
    log.info("removed %d exact/near-duplicate images", before - len(df))

    if max_per_class:
        df = pd.concat([g.sample(n=min(len(g), max_per_class), random_state=seed)
                        for _, g in df.groupby("label")])
        log.info("capped each class at %d images", max_per_class)

    df = df.reset_index(drop=True)
    df["label_idx"] = df["label"].map(CLASS_TO_IDX)
    return df


def make_splits(df: pd.DataFrame, seed: int, val_frac: float = 0.1, test_frac: float = 0.1) -> pd.DataFrame:
    """Group-aware, stratified train/val/test split. Augmentation happens later, train-only."""
    df = df.copy()
    idx = np.arange(len(df))
    k_test = max(2, round(1 / test_frac))
    sg = StratifiedGroupKFold(n_splits=k_test, shuffle=True, random_state=seed)
    rest, test = next(sg.split(idx, df["label_idx"], df["group"]))
    k_val = max(2, round((1 - test_frac) / val_frac))
    sub = df.iloc[rest]
    sg2 = StratifiedGroupKFold(n_splits=k_val, shuffle=True, random_state=seed + 1)
    tr_rel, va_rel = next(sg2.split(np.arange(len(sub)), sub["label_idx"], sub["group"]))
    df["split"] = "train"
    df.loc[df.index[rest[va_rel]], "split"] = "val"
    df.loc[df.index[test], "split"] = "test"

    # leakage assertions
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        ga, gb = set(df[df.split == a].group), set(df[df.split == b].group)
        ha, hb = set(df[df.split == a].dhash), set(df[df.split == b].dhash)
        assert not ga & gb, f"patient/group leakage between {a} and {b}"
        assert not ha & hb, f"duplicate-image leakage between {a} and {b}"
    return df


def print_audit(df: pd.DataFrame) -> None:
    print("\n=== Class x split ===")
    print(pd.crosstab(df["label"], df["split"]) if "split" in df else df["label"].value_counts())
    print("\n=== Class x source ===")
    ct = pd.crosstab(df["label"], df["source"])
    print(ct)
    for cls in CLASSES:
        if cls in ct.index and (ct.loc[cls] > 0).sum() == 1:
            only = ct.columns[(ct.loc[cls] > 0).values][0]
            print(f"[WARN] '{cls}' comes from a single source ({only}). The model may learn the "
                  f"source/scanner instead of the disease - read per-source results carefully.")
    print("\n=== Image size summary (px) ===")
    print(df.groupby("source")[["width", "height"]].agg(["min", "median", "max"]))
    print()


# --------------------------------------------------------------------------- #
# Preprocessing, CLAHE & augmentation  (report sec. 4.3 / 7.3)
# --------------------------------------------------------------------------- #
@dataclass
class PreprocConfig:
    img_size: int = 224
    clahe: bool = False          # deterministic CLAHE on EVERY image (train/val/test/inference)
    clip_limit: float = 2.0
    tile_grid: int = 8
    augment: bool = False        # train-only augmentation
    aug_random_clahe_p: float = 0.0   # train-only: P(apply CLAHE) when clahe=False
    aug_rotate_deg: float = 7.0
    aug_scale: float = 0.05
    aug_shift: float = 0.03
    aug_alpha: Tuple[float, float] = (0.9, 1.1)   # contrast gain
    aug_beta: Tuple[float, float] = (-10, 10)     # brightness offset (0-255 scale)


PRESETS: Dict[str, Dict] = {
    # (1) plain resize + normalisation
    "baseline": dict(clahe=False, augment=False),
    # (2) CLAHE-preprocessed images everywhere
    "clahe": dict(clahe=True, augment=False),
    # (3) training-time random choice original/CLAHE + mild brightness/contrast + small geometry
    "aug_mix": dict(clahe=False, augment=True, aug_random_clahe_p=0.5),
    # CLAHE everywhere + mild geometric/brightness augmentation
    "clahe_aug": dict(clahe=True, augment=True, aug_random_clahe_p=0.0),
}


def make_cfg(preset: str, img_size: int, clip: float, tile: int) -> PreprocConfig:
    return PreprocConfig(img_size=img_size, clip_limit=clip, tile_grid=tile, **PRESETS[preset])


def apply_clahe(gray: np.ndarray, clip_limit: float, tile_grid: int) -> np.ndarray:
    return cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid)).apply(gray)


def preprocess_gray(gray: np.ndarray, cfg: PreprocConfig) -> np.ndarray:
    """Shared by training, evaluation and inference (FR2): resize -> optional CLAHE."""
    g = cv2.resize(gray, (cfg.img_size, cfg.img_size), interpolation=cv2.INTER_AREA)
    if cfg.clahe:
        g = apply_clahe(g, cfg.clip_limit, cfg.tile_grid)
    return g


def augment_gray(g: np.ndarray, cfg: PreprocConfig) -> np.ndarray:
    """Conservative, label-preserving, TRAIN-ONLY augmentation (no flips: heart side matters)."""
    if cfg.aug_random_clahe_p > 0 and random.random() < cfg.aug_random_clahe_p:
        g = apply_clahe(g, random.uniform(1.5, 3.0), cfg.tile_grid)
    s = g.shape[0]
    ang = random.uniform(-cfg.aug_rotate_deg, cfg.aug_rotate_deg)
    sc = 1.0 + random.uniform(-cfg.aug_scale, cfg.aug_scale)
    M = cv2.getRotationMatrix2D((s / 2, s / 2), ang, sc)
    M[:, 2] += [random.uniform(-cfg.aug_shift, cfg.aug_shift) * s for _ in range(2)]
    g = cv2.warpAffine(g, M, (s, s), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    a = random.uniform(*cfg.aug_alpha)
    b = random.uniform(*cfg.aug_beta)
    return cv2.convertScaleAbs(g, alpha=a, beta=b)


def to_tensor(g: np.ndarray) -> torch.Tensor:
    x = np.repeat((g.astype(np.float32) / 255.0)[..., None], 3, axis=2)
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))


class XrayDataset(Dataset):
    def __init__(self, df: pd.DataFrame, cfg: PreprocConfig, train: bool = False):
        self.paths = df["path"].tolist()
        self.labels = df["label_idx"].astype(int).tolist()
        self.cfg, self.train = cfg, train

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        gray = read_gray(self.paths[i])
        if gray is None:
            gray = np.zeros((self.cfg.img_size, self.cfg.img_size), np.uint8)
            log.warning("unreadable image replaced by blank: %s", self.paths[i])
        g = preprocess_gray(gray, self.cfg)
        if self.train and self.cfg.augment:
            g = augment_gray(g, self.cfg)
        return to_tensor(g), self.labels[i]


def make_loader(df, cfg, train, batch_size, workers, sampler=None, augment_train=True) -> DataLoader:
    ds = XrayDataset(df, cfg, train=train and augment_train)
    return DataLoader(ds, batch_size=batch_size, shuffle=(train and sampler is None), sampler=sampler,
                      num_workers=workers, pin_memory=torch.cuda.is_available(),
                      persistent_workers=workers > 0, drop_last=False)


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
def build_densenet(num_classes: int = len(CLASSES), pretrained: bool = True, dropout: float = 0.3):
    weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
    m = models.densenet121(weights=weights)
    m.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(m.classifier.in_features, num_classes))
    return m


def pooled_features(model: nn.Module, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (last conv feature map, 1024-d GAP vector); non-inplace ReLU so Grad-CAM works."""
    fmap = model.features(x)
    pooled = F.adaptive_avg_pool2d(F.relu(fmap), 1).flatten(1)
    return fmap, pooled


def set_trainable(model: nn.Module, stage: str) -> None:
    for p in model.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = True
    if stage == "full":
        for p in model.parameters():
            p.requires_grad = True
    elif stage == "partial":
        for name in ("denseblock4", "norm5", "transition3", "denseblock3"):
            for p in getattr(model.features, name).parameters():
                p.requires_grad = True


def set_train_mode(model: nn.Module) -> None:
    """Train mode, but frozen blocks keep BatchNorm statistics fixed."""
    model.train()
    for _, mod in model.features.named_children():
        if not any(p.requires_grad for p in mod.parameters()):
            mod.eval()


# --------------------------------------------------------------------------- #
# Metrics (report sec. 7.6)
# --------------------------------------------------------------------------- #
def compute_metrics(y_true, y_pred, scores=None) -> Dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    K = len(CLASSES)
    labels = list(range(K))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    spec = []
    for k in labels:
        fp = cm[:, k].sum() - cm[k, k]
        tn = cm.sum() - cm[k, :].sum() - cm[:, k].sum() + cm[k, k]
        spec.append(float(tn / (tn + fp)) if (tn + fp) else float("nan"))
    out = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "per_class": {
            CLASSES[k]: {"precision": float(p[k]), "recall_sensitivity": float(r[k]),
                         "specificity": spec[k], "f1": float(f[k]), "support": int(s[k])}
            for k in labels
        },
        "confusion_matrix": cm.tolist(),
    }
    if scores is not None:
        aucs = {}
        for k in labels:
            yk = (y_true == k).astype(int)
            if 0 < yk.sum() < len(yk):
                aucs[CLASSES[k]] = float(roc_auc_score(yk, scores[:, k]))
        out["roc_auc_ovr"] = aucs
        out["roc_auc_ovr_macro"] = float(np.mean(list(aucs.values()))) if aucs else float("nan")
    return out


def plot_confusion(cm, title: str, path: Path) -> None:
    cm = np.asarray(cm)
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=30, ha="right")
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, f"{cm[i, j]}\n({norm[i, j]:.0%})", ha="center", va="center",
                    color="white" if norm[i, j] > 0.5 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def softmax_np(z: np.ndarray) -> np.ndarray:
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


# --------------------------------------------------------------------------- #
# Run-directory helpers
# --------------------------------------------------------------------------- #
def run_dir(args, preset: str, model: str) -> Path:
    return Path(args.workdir) / "runs" / preset / model


def load_manifest(args) -> pd.DataFrame:
    p = Path(args.workdir) / "manifest.csv"
    if not p.exists():
        raise SystemExit(f"{p} not found - run the 'prepare' command first.")
    return pd.read_csv(p)


# --------------------------------------------------------------------------- #
# Model A : DenseNet121 transfer learning (report sec. 7.4)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def predict_probs(model, loader, device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    P, Y = [], []
    for x, y in loader:
        logits = model(x.to(device, non_blocking=True))
        P.append(F.softmax(logits.float(), 1).cpu().numpy())
        Y.append(y.numpy())
    return np.concatenate(P), np.concatenate(Y)


def train_cnn(args, preset: Optional[str] = None) -> Path:
    preset = preset or args.preset
    seed_everything(args.seed)
    device = pick_device(args.device)
    df = load_manifest(args)
    cfg = make_cfg(preset, args.img_size, args.clip_limit, args.tile_grid)
    tr, va = df[df.split == "train"], df[df.split == "val"]
    out = run_dir(args, preset, "cnn")
    out.mkdir(parents=True, exist_ok=True)

    counts = np.bincount(tr["label_idx"], minlength=len(CLASSES)).astype(float)
    class_w = counts.sum() / (len(CLASSES) * np.maximum(counts, 1))
    sampler = None
    if args.balanced_sampler:
        sw = class_w[tr["label_idx"].values]
        sampler = WeightedRandomSampler(sw, num_samples=len(tr), replacement=True)
    crit = nn.CrossEntropyLoss(
        weight=None if args.balanced_sampler else torch.tensor(class_w, dtype=torch.float32, device=device),
        label_smoothing=0.05)
    log.info("[cnn:%s] class weights %s", preset, dict(zip(CLASSES, np.round(class_w, 2))))

    tl = make_loader(tr, cfg, True, args.batch_size, args.num_workers, sampler)
    vl = make_loader(va, cfg, False, args.batch_size, args.num_workers)
    model = build_densenet(pretrained=not args.no_pretrained).to(device)
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_f1, best_state, bad, history = -1.0, None, 0, []
    stages = [("head", args.epochs_head, args.lr_head),
              ("full" if args.full_finetune else "partial", args.epochs_ft, args.lr_ft)]
    for stage, n_ep, lr in stages:
        if n_ep <= 0:
            continue
        set_trainable(model, stage)
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=2)
        bad = 0
        for ep in range(1, n_ep + 1):
            set_train_mode(model)
            t0, run_loss, seen = time.time(), 0.0, 0
            for x, y in tl:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    loss = crit(model(x), y)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                run_loss += loss.item() * len(y)
                seen += len(y)
            P, Y = predict_probs(model, vl, device)
            f1 = f1_score(Y, P.argmax(1), average="macro", labels=list(range(len(CLASSES))), zero_division=0)
            sched.step(f1)
            history.append({"stage": stage, "epoch": ep, "train_loss": run_loss / max(seen, 1),
                            "val_macro_f1": float(f1), "val_acc": float(accuracy_score(Y, P.argmax(1)))})
            log.info("[cnn:%s] %-7s ep %2d/%d loss %.4f | val macro-F1 %.4f | %.0fs",
                     preset, stage, ep, n_ep, run_loss / max(seen, 1), f1, time.time() - t0)
            if f1 > best_f1:
                best_f1, bad = f1, 0
                best_state = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
            else:
                bad += 1
                if bad >= args.patience:
                    log.info("[cnn:%s] early stopping (stage %s)", preset, stage)
                    break
        if best_state is not None:
            model.load_state_dict(best_state)

    ckpt = {"state_dict": best_state, "classes": CLASSES, "preset": preset, "preproc": asdict(cfg),
            "version": MODEL_VERSION, "seed": args.seed, "best_val_macro_f1": best_f1,
            "history": history, "versions": env_versions()}
    torch.save(ckpt, out / "model.pt")
    save_json({k: v for k, v in ckpt.items() if k != "state_dict"}, out / "run_config.json")
    log.info("[cnn:%s] saved %s (best val macro-F1 %.4f)", preset, out / "model.pt", best_f1)
    return out / "model.pt"


def load_cnn(ckpt_path: Path, device) -> Tuple[nn.Module, PreprocConfig, dict]:
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    m = build_densenet(pretrained=False)
    m.load_state_dict(ck["state_dict"])
    cfg = PreprocConfig(**{**ck["preproc"], "aug_alpha": tuple(ck["preproc"]["aug_alpha"]),
                           "aug_beta": tuple(ck["preproc"]["aug_beta"])})
    return m.to(device).eval(), cfg, ck


# --------------------------------------------------------------------------- #
# Model B : deep features + SVC (report sec. 7.5)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def extract_features(model, loader, device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    X, Y = [], []
    for x, y in loader:
        _, pooled = pooled_features(model, x.to(device, non_blocking=True))
        X.append(pooled.float().cpu().numpy())
        Y.append(y.numpy())
    return np.concatenate(X), np.concatenate(Y)


def train_svc(args, preset: Optional[str] = None) -> Path:
    preset = preset or args.preset
    seed_everything(args.seed)
    device = pick_device(args.device)
    df = load_manifest(args)
    cfg = make_cfg(preset, args.img_size, args.clip_limit, args.tile_grid)
    tr, va = df[df.split == "train"], df[df.split == "val"]
    out = run_dir(args, preset, "svc")
    out.mkdir(parents=True, exist_ok=True)

    if args.backbone == "finetuned":
        backbone, _, _ = load_cnn(run_dir(args, preset, "cnn") / "model.pt", device)
    else:  # frozen ImageNet DenseNet121 as a pure feature extractor
        backbone = build_densenet(pretrained=not args.no_pretrained).to(device).eval()
    torch.save(backbone.state_dict(), out / "backbone.pt")

    # features: deterministic pass + optional train-only augmented copies
    trl = make_loader(tr, cfg, False, args.batch_size, args.num_workers)
    Xtr, ytr = extract_features(backbone, trl, device)
    if args.svc_aug_copies > 0 and cfg.augment:
        for c in range(args.svc_aug_copies):
            augl = make_loader(tr, cfg, True, args.batch_size, args.num_workers)
            Xa, ya = extract_features(backbone, augl, device)
            Xtr, ytr = np.vstack([Xtr, Xa]), np.concatenate([ytr, ya])
            log.info("[svc:%s] added augmented copy %d/%d", preset, c + 1, args.svc_aug_copies)
    Xva, yva = extract_features(backbone, make_loader(va, cfg, False, args.batch_size, args.num_workers), device)
    log.info("[svc:%s] features train %s  val %s", preset, Xtr.shape, Xva.shape)

    scaler = StandardScaler().fit(Xtr)               # statistics from TRAIN only
    Ztr, Zva = scaler.transform(Xtr), scaler.transform(Xva)
    grid = [("linear", c, "scale") for c in (0.01, 0.1, 1.0)]
    grid += [("rbf", c, "scale") for c in (1.0, 10.0, 100.0)]
    best, results = None, []
    for kernel, C, gamma in grid:                    # model selection on VALIDATION only
        svc = SVC(kernel=kernel, C=C, gamma=gamma, class_weight="balanced",
                  decision_function_shape="ovr", random_state=args.seed).fit(Ztr, ytr)
        f1 = f1_score(yva, svc.predict(Zva), average="macro", zero_division=0)
        results.append({"kernel": kernel, "C": C, "gamma": gamma, "val_macro_f1": float(f1)})
        log.info("[svc:%s] kernel=%-6s C=%-6g val macro-F1 %.4f", preset, kernel, C, f1)
        if best is None or f1 > best[0]:
            best = (f1, kernel, C, gamma)
    _, kernel, C, gamma = best
    final = Pipeline([("scaler", StandardScaler()),
                      ("svc", SVC(kernel=kernel, C=C, gamma=gamma, class_weight="balanced",
                                  decision_function_shape="ovr", probability=args.svc_probability,
                                  random_state=args.seed))]).fit(Xtr, ytr)
    bundle = {"pipeline": final, "classes": CLASSES, "preset": preset, "preproc": asdict(cfg),
              "backbone": args.backbone, "best": {"kernel": kernel, "C": C, "gamma": gamma},
              "search": results, "version": MODEL_VERSION, "probability_calibrated": args.svc_probability,
              "versions": env_versions()}
    joblib.dump(bundle, out / "svc.joblib")
    save_json({k: v for k, v in bundle.items() if k != "pipeline"}, out / "run_config.json")
    log.info("[svc:%s] best %s (val macro-F1 %.4f) -> %s", preset, bundle["best"], best[0], out / "svc.joblib")
    return out / "svc.joblib"


def load_svc(svc_dir: Path, device):
    bundle = joblib.load(svc_dir / "svc.joblib")
    bb = build_densenet(pretrained=False)
    bb.load_state_dict(torch.load(svc_dir / "backbone.pt", map_location="cpu"))
    cfg = PreprocConfig(**{**bundle["preproc"], "aug_alpha": tuple(bundle["preproc"]["aug_alpha"]),
                           "aug_beta": tuple(bundle["preproc"]["aug_beta"])})
    return bb.to(device).eval(), bundle, cfg


# --------------------------------------------------------------------------- #
# Evaluation on the held-out TEST split
# --------------------------------------------------------------------------- #
def evaluate(args, model_kind: Optional[str] = None, preset: Optional[str] = None) -> Dict:
    model_kind, preset = model_kind or args.model, preset or args.preset
    device = pick_device(args.device)
    df = load_manifest(args)
    te = df[df.split == "test"].reset_index(drop=True)
    out = run_dir(args, preset, model_kind)

    if model_kind == "cnn":
        model, cfg, _ = load_cnn(out / "model.pt", device)
        scores, y = predict_probs(model, make_loader(te, cfg, False, args.batch_size, args.num_workers), device)
        pred = scores.argmax(1)
    else:
        bb, bundle, cfg = load_svc(out, device)
        X, y = extract_features(bb, make_loader(te, cfg, False, args.batch_size, args.num_workers), device)
        pipe = bundle["pipeline"]
        pred, scores = pipe.predict(X), pipe.decision_function(X)   # margins, NOT probabilities

    m = compute_metrics(y, pred, scores)
    m["model"], m["preset"], m["version"] = model_kind, preset, MODEL_VERSION
    m["per_source"] = {}
    for src, g in te.groupby("source"):
        ii = g.index.values
        m["per_source"][src] = {"n": int(len(ii)), "accuracy": float(accuracy_score(y[ii], pred[ii])),
                                "macro_f1": float(f1_score(y[ii], pred[ii], average="macro", zero_division=0)),
                                "classes_present": sorted({CLASSES[k] for k in y[ii]})}
    save_json(m, out / "test_metrics.json")
    plot_confusion(m["confusion_matrix"], f"{model_kind.upper()} / {preset}", out / "confusion_matrix.png")
    pr = te[["path", "source", "label"]].copy()
    pr["pred"] = [CLASSES[k] for k in pred]
    pr["correct"] = pr["label"] == pr["pred"]
    for k, c in enumerate(CLASSES):
        pr[f"score_{c}"] = scores[:, k]
    pr.to_csv(out / "predictions.csv", index=False)           # for false-positive / false-negative review
    log.info("[eval:%s/%s] acc %.4f | macro-F1 %.4f | macro AUC %.4f", model_kind, preset,
             m["accuracy"], m["macro_f1"], m.get("roc_auc_ovr_macro", float("nan")))
    return m


# --------------------------------------------------------------------------- #
# Grad-CAM (CNN branch only)  (report sec. 4.5 / 7.7)
# --------------------------------------------------------------------------- #
def gradcam(model: nn.Module, x: torch.Tensor, target: Optional[int] = None) -> Tuple[np.ndarray, int, np.ndarray]:
    """Grad-CAM on the last DenseNet feature map. Returns (cam in [0,1], class idx, probs)."""
    model.eval()
    x = x.clone().requires_grad_(True)
    with torch.enable_grad():
        fmap, pooled = pooled_features(model, x)
        fmap.retain_grad()
        logits = model.classifier(pooled)
        probs = F.softmax(logits, 1)[0].detach().cpu().numpy()
        idx = int(logits.argmax(1).item()) if target is None else int(target)
        model.zero_grad(set_to_none=True)
        logits[0, idx].backward()
    w = fmap.grad.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * fmap).sum(1)).squeeze(0).detach().cpu().numpy()
    cam = cam - cam.min()
    cam = cam / cam.max() if cam.max() > 0 else cam
    return cam, idx, probs


def overlay_cam(gray_224: np.ndarray, cam: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    h, w = gray_224.shape
    heat = cv2.applyColorMap((cv2.resize(cam, (w, h)) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    base = cv2.cvtColor(gray_224, cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(base, 1 - alpha, heat, alpha, 0)


def png_b64(img_bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img_bgr)
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


def cmd_gradcam(args) -> None:
    device = pick_device(args.device)
    model, cfg, _ = load_cnn(run_dir(args, args.preset, "cnn") / "model.pt", device)
    gray = read_gray(args.image)
    if gray is None:
        raise SystemExit(f"Cannot read image: {args.image}")
    g = preprocess_gray(gray, cfg)
    cam, idx, probs = gradcam(model, to_tensor(g).unsqueeze(0).to(device))
    ov = overlay_cam(g, cam)
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].imshow(g, cmap="gray"); ax[0].set_title("Input (preprocessed)")
    ax[1].imshow(cv2.cvtColor(ov, cv2.COLOR_BGR2RGB)); ax[1].set_title(f"Grad-CAM: {CLASSES[idx]} ({probs[idx]:.0%})")
    for a in ax:
        a.axis("off")
    fig.suptitle("Research prototype - not for clinical use", fontsize=8, color="firebrick")
    out = Path(args.out)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}\n{GRADCAM_NOTE}")


# --------------------------------------------------------------------------- #
# CLAHE / augmentation visual check  (report: "team will check some examples")
# --------------------------------------------------------------------------- #
def cmd_clahe_demo(args) -> None:
    df = load_manifest(args)
    rng = np.random.RandomState(args.seed)
    picks = []
    for c in CLASSES:
        sub = df[df.label == c]
        if len(sub):
            picks.append(sub.iloc[rng.randint(len(sub))])
    cols = ["Original", "CLAHE clip=2", "CLAHE clip=4", "Augmented #1", "Augmented #2"]
    fig, axes = plt.subplots(len(picks), len(cols), figsize=(2.6 * len(cols), 2.6 * len(picks)))
    axes = np.atleast_2d(axes)
    base = PreprocConfig(img_size=args.img_size, tile_grid=args.tile_grid, augment=True, aug_random_clahe_p=0.5)
    random.seed(args.seed)
    for r, row in enumerate(picks):
        g = preprocess_gray(read_gray(row.path), PreprocConfig(img_size=args.img_size))
        imgs = [g, apply_clahe(g, 2.0, args.tile_grid), apply_clahe(g, 4.0, args.tile_grid),
                augment_gray(g, base), augment_gray(g, base)]
        for c, im in enumerate(imgs):
            axes[r, c].imshow(im, cmap="gray", vmin=0, vmax=255)
            axes[r, c].axis("off")
            if r == 0:
                axes[r, c].set_title(cols[c], fontsize=9)
        axes[r, 0].text(-0.05, 0.5, row.label, transform=axes[r, 0].transAxes, rotation=90,
                        va="center", ha="right", fontsize=9)
    fig.tight_layout()
    out = Path(args.workdir) / "clahe_augmentation_examples.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}  (inspect: does CLAHE amplify noise / create artefacts? record clip & tile values)")


# --------------------------------------------------------------------------- #
# Full experiment grid: presets x {CNN head, SVC}
# --------------------------------------------------------------------------- #
def cmd_run_all(args) -> None:
    rows = []
    for preset in args.presets:
        log.info("================ preset: %s ================", preset)
        train_cnn(args, preset)
        train_svc(args, preset)
        for kind in ("cnn", "svc"):
            m = evaluate(args, kind, preset)
            rows.append({"preset": preset, "model": kind, "accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
                         "macro_auc_ovr": m.get("roc_auc_ovr_macro"),
                         **{f"f1_{c}": m["per_class"][c]["f1"] for c in CLASSES}})
    res = pd.DataFrame(rows)
    out = Path(args.workdir) / "comparison.csv"
    res.to_csv(out, index=False)
    print("\n" + res.round(4).to_string(index=False))
    print(f"\nsaved {out}. Report only these measured values; do not quote published accuracies as targets.")


# --------------------------------------------------------------------------- #
# Inference service + FastAPI (report FR1, FR2, FR4-FR7, FR9)
# --------------------------------------------------------------------------- #
class InferenceError(Exception):
    def __init__(self, status: int, msg: str):
        super().__init__(msg)
        self.status, self.msg = status, msg


class InferenceService:
    def __init__(self, workdir: Path, preset: str, device: torch.device):
        self.device, self.preset = device, preset
        base = Path(workdir) / "runs" / preset
        self.cnn = self.cnn_cfg = self.svc_bb = self.svc_bundle = self.svc_cfg = None
        if (base / "cnn" / "model.pt").exists():
            self.cnn, self.cnn_cfg, _ = load_cnn(base / "cnn" / "model.pt", device)
        if (base / "svc" / "svc.joblib").exists():
            self.svc_bb, self.svc_bundle, self.svc_cfg = load_svc(base / "svc", device)
        if self.cnn is None and self.svc_bundle is None:
            log.warning("no trained weights found under %s", base)

    @property
    def branches(self) -> List[str]:
        return [b for b, ok in (("cnn", self.cnn is not None), ("svc", self.svc_bundle is not None)) if ok]

    def validate(self, filename: str, data: bytes) -> np.ndarray:
        ext = Path(filename or "").suffix.lower()
        if ext not in IMAGE_EXTS:
            raise InferenceError(415, f"Unsupported file type '{ext}'. Allowed: {sorted(IMAGE_EXTS)}")
        if len(data) == 0 or len(data) > MAX_UPLOAD_BYTES:
            raise InferenceError(413, f"File must be 1 byte - {MAX_UPLOAD_BYTES // 2**20} MB.")
        gray = decode_gray(np.frombuffer(data, np.uint8))
        if gray is None:
            raise InferenceError(400, "Image could not be decoded (corrupt or not an image).")
        if min(gray.shape) < 64:
            raise InferenceError(400, "Image is too small (min 64 px per side).")
        return gray

    @torch.no_grad()
    def _svc_predict(self, g: np.ndarray) -> Dict:
        x = to_tensor(g).unsqueeze(0).to(self.device)
        _, pooled = pooled_features(self.svc_bb, x)
        pipe = self.svc_bundle["pipeline"]
        feat = pooled.cpu().numpy()
        idx = int(pipe.predict(feat)[0])
        res = {"predicted_class": CLASSES[idx], "branch": "svc"}
        if self.svc_bundle.get("probability_calibrated"):
            res["class_probabilities"] = dict(zip(CLASSES, map(float, pipe.predict_proba(feat)[0])))
        else:
            res["decision_scores"] = dict(zip(CLASSES, map(float, pipe.decision_function(feat)[0])))
            res["score_note"] = "SVC margins are not probabilities. No Grad-CAM exists for the SVC branch."
        return res

    def predict(self, filename: str, data: bytes, branch: str = "cnn") -> Dict:
        if branch not in ("cnn", "svc"):
            raise InferenceError(400, "branch must be 'cnn' or 'svc'")
        if branch not in self.branches:
            raise InferenceError(503, f"Model weights for branch '{branch}' are not available.")
        gray = self.validate(filename, data)
        t0 = time.time()
        try:
            if branch == "cnn":
                g = preprocess_gray(gray, self.cnn_cfg)
                cam, idx, probs = gradcam(self.cnn, to_tensor(g).unsqueeze(0).to(self.device))
                res = {"predicted_class": CLASSES[idx], "branch": "cnn",
                       "class_scores": dict(zip(CLASSES, map(float, probs))),
                       "score_note": "Softmax scores are uncalibrated model outputs, not clinical probabilities.",
                       "gradcam_overlay": png_b64(overlay_cam(g, cam)), "gradcam_note": GRADCAM_NOTE}
            else:
                res = self._svc_predict(preprocess_gray(gray, self.svc_cfg))
        except InferenceError:
            raise
        except Exception as exc:  # FR9: inference failure
            log.exception("inference failed")
            raise InferenceError(500, f"Inference error: {exc}")
        res.update({"model_version": MODEL_VERSION, "preset": self.preset,
                    "latency_ms": round((time.time() - t0) * 1000, 1), "disclaimer": DISCLAIMER})
        return res


def create_app(workdir: Path, preset: str, device: torch.device):
    from fastapi import FastAPI, File, Form, UploadFile
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse

    app = FastAPI(title="Radiology AI Assistant (research prototype)", version=MODEL_VERSION)
    app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:3000", "http://localhost:5173"],
                       allow_methods=["*"], allow_headers=["*"])
    service = InferenceService(workdir, preset, device)

    @app.exception_handler(InferenceError)
    async def _err(_, exc: InferenceError):
        return JSONResponse(status_code=exc.status, content={"error": exc.msg, "disclaimer": DISCLAIMER})

    @app.get("/health")
    def health():
        return {"status": "ok", "model_version": MODEL_VERSION, "preset": preset,
                "available_branches": service.branches, "classes": CLASSES, "disclaimer": DISCLAIMER}

    @app.post("/predict")
    async def predict(file: UploadFile = File(...), branch: str = Form("cnn")):
        data = await file.read(MAX_UPLOAD_BYTES + 1)
        return service.predict(file.filename, data, branch)

    return app


def cmd_serve(args) -> None:
    import uvicorn
    uvicorn.run(create_app(Path(args.workdir), args.preset, pick_device(args.device)),
                host=args.host, port=args.port)


# --------------------------------------------------------------------------- #
# prepare / CLI
# --------------------------------------------------------------------------- #
def cmd_prepare(args) -> None:
    seed_everything(args.seed)
    overrides = dict(s.split("=", 1) for s in (args.source or []))
    df = build_manifest(Path(args.data_root), overrides, args.sources, args.max_per_class, args.seed, args.num_workers)
    df = make_splits(df, args.seed)
    out = Path(args.workdir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "manifest.csv", index=False)
    save_json({"seed": args.seed, "sources": sorted(df.source.unique()), "n_images": len(df),
               "label_mapping_note": "Lung_Opacity excluded; Kermany PNEUMONIA includes bacterial+viral.",
               "versions": env_versions()}, out / "dataset_info.json")
    print_audit(df)
    print(f"saved {out / 'manifest.csv'} (group-aware split, de-duplicated, no leakage across splits)")


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--workdir", default="./workdir")
    common.add_argument("--seed", type=int, default=42)
    common.add_argument("--device", default="auto")
    common.add_argument("--img-size", type=int, default=224)
    common.add_argument("--clip-limit", type=float, default=2.0, help="CLAHE clip limit")
    common.add_argument("--tile-grid", type=int, default=8, help="CLAHE tile grid size (NxN)")
    common.add_argument("--batch-size", type=int, default=32)
    common.add_argument("--num-workers", type=int, default=4)
    common.add_argument("--no-pretrained", action="store_true", help="skip ImageNet weights (offline/smoke tests)")

    train = argparse.ArgumentParser(add_help=False)
    train.add_argument("--epochs-head", type=int, default=3)
    train.add_argument("--epochs-ft", type=int, default=12)
    train.add_argument("--lr-head", type=float, default=1e-3)
    train.add_argument("--lr-ft", type=float, default=1e-4)
    train.add_argument("--patience", type=int, default=4)
    train.add_argument("--full-finetune", action="store_true")
    train.add_argument("--balanced-sampler", action="store_true", help="use sampler instead of class-weighted loss")
    train.add_argument("--backbone", choices=["imagenet", "finetuned"], default="imagenet",
                       help="feature extractor for the SVC branch")
    train.add_argument("--svc-aug-copies", type=int, default=0, help="train-only augmented feature copies")
    train.add_argument("--svc-probability", action="store_true", help="Platt-calibrated SVC probabilities (slower)")

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare", parents=[common], help="build manifest, de-duplicate, split, audit")
    p.add_argument("--data-root", required=True)
    p.add_argument("--source", action="append", metavar="NAME=PATH", help="override a source folder")
    p.add_argument("--sources", nargs="*", choices=list(SOURCES), help="restrict to these sources")
    p.add_argument("--max-per-class", type=int, default=None, help="cap images per class (balance / speed)")

    sub.add_parser("audit", parents=[common], help="print dataset audit from manifest")
    sub.add_parser("clahe-demo", parents=[common], help="save CLAHE / augmentation example grid")

    for name in ("train-cnn", "train-svc"):
        p = sub.add_parser(name, parents=[common, train])
        p.add_argument("--preset", choices=list(PRESETS), default="clahe_aug")

    p = sub.add_parser("evaluate", parents=[common], help="evaluate on held-out TEST split")
    p.add_argument("--model", choices=["cnn", "svc"], required=True)
    p.add_argument("--preset", choices=list(PRESETS), default="clahe_aug")

    p = sub.add_parser("run-all", parents=[common, train], help="train + evaluate every preset, CNN and SVC")
    p.add_argument("--presets", nargs="+", choices=list(PRESETS), default=list(PRESETS))

    p = sub.add_parser("gradcam", parents=[common], help="Grad-CAM figure for one image (CNN branch)")
    p.add_argument("--preset", choices=list(PRESETS), default="clahe_aug")
    p.add_argument("--image", required=True)
    p.add_argument("--out", default="gradcam_output.png")

    p = sub.add_parser("serve", parents=[common], help="start FastAPI inference service")
    p.add_argument("--preset", choices=list(PRESETS), default="clahe_aug")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    return ap


def main(argv: Optional[List[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    args = build_parser().parse_args(argv)
    {
        "prepare": cmd_prepare,
        "audit": lambda a: print_audit(load_manifest(a)),
        "clahe-demo": cmd_clahe_demo,
        "train-cnn": train_cnn,
        "train-svc": train_svc,
        "evaluate": evaluate,
        "run-all": cmd_run_all,
        "gradcam": cmd_gradcam,
        "serve": cmd_serve,
    }[args.cmd](args)


if __name__ == "__main__":
    main()
