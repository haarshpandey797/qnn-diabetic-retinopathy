"""Loading and preprocessing of the IDRiD disease-grading images and labels."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

GRADE_NAMES = [
    "No DR",
    "Mild NPDR",
    "Moderate NPDR",
    "Severe NPDR",
    "Proliferative DR",
]
NUM_CLASSES = len(GRADE_NAMES)

IMAGE_COLUMN = "Image name"
LABEL_COLUMN = "Retinopathy grade"


@dataclass
class Split:
    """One split of the dataset."""

    images: np.ndarray  # (N, size, size, 3), float32 in [0, 1]
    labels: np.ndarray  # (N,), int DR grade 0-4
    ids: list[str]


def crop_to_fundus(img: Image.Image, threshold: int = 10, min_fraction: float = 0.05) -> Image.Image:
    """Crop the black background around the circular fundus.

    IDRiD images are 4288x2848 with wide black margins. Cropping first means the
    downscaled image spends its few pixels on the retina instead of the border.
    A row or column counts as fundus only if at least `min_fraction` of its
    pixels are brighter than `threshold`, so stray bright border pixels from
    JPEG noise do not widen the crop.
    """
    gray = np.asarray(img.convert("L"))
    bright = gray > threshold
    rows = np.flatnonzero(bright.mean(axis=1) >= min_fraction)
    cols = np.flatnonzero(bright.mean(axis=0) >= min_fraction)
    if rows.size == 0 or cols.size == 0:
        return img
    return img.crop((int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1))


def pad_to_square(img: Image.Image) -> Image.Image:
    """Pad with black to a square so resizing does not squash the round fundus."""
    width, height = img.size
    side = max(width, height)
    if width == height:
        return img
    canvas = Image.new("RGB", (side, side))
    canvas.paste(img, ((side - width) // 2, (side - height) // 2))
    return canvas


def load_image(path: str | Path, size: int) -> np.ndarray:
    """Load one fundus image as a (size, size, 3) float32 array in [0, 1]."""
    with Image.open(path) as img:
        # Let the JPEG decoder downscale (up to 1/8) while staying at least twice
        # the target size; decoding a 4288x2848 image at full size is wasted work.
        img.draft("RGB", (size * 2, size * 2))
        img = pad_to_square(crop_to_fundus(img.convert("RGB")))
        img = img.resize((size, size), resample=Image.Resampling.BILINEAR)
        return np.asarray(img, dtype=np.float32) / 255.0


def read_labels(csv_path: str | Path) -> pd.DataFrame:
    """Read an IDRiD grading CSV and keep only the image name and DR grade.

    The official CSVs have a trailing space in one header and several empty
    "Unnamed" columns, so headers are stripped and extra columns dropped.
    """
    df = pd.read_csv(csv_path)
    df.columns = [str(c).strip() for c in df.columns]
    missing = {IMAGE_COLUMN, LABEL_COLUMN} - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path}: missing column(s) {sorted(missing)}")

    df = df[[IMAGE_COLUMN, LABEL_COLUMN]].dropna()
    df[IMAGE_COLUMN] = df[IMAGE_COLUMN].astype(str).str.strip()
    df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(int)

    out_of_range = ~df[LABEL_COLUMN].between(0, NUM_CLASSES - 1)
    if out_of_range.any():
        bad = sorted(df.loc[out_of_range, LABEL_COLUMN].unique())
        raise ValueError(f"{csv_path}: unexpected grade value(s) {bad}")
    return df.reset_index(drop=True)


def load_split(csv_path: str | Path, image_dir: str | Path, size: int = 28) -> Split:
    """Load every image listed in `csv_path` from `image_dir`."""
    df = read_labels(csv_path)
    image_dir = Path(image_dir)
    images = np.empty((len(df), size, size, 3), dtype=np.float32)
    for i, name in enumerate(df[IMAGE_COLUMN]):
        path = image_dir / f"{name}.jpg"
        if not path.exists():
            raise FileNotFoundError(f"Image listed in {csv_path} not found: {path}")
        images[i] = load_image(path, size)
    return Split(images=images, labels=df[LABEL_COLUMN].to_numpy(), ids=df[IMAGE_COLUMN].tolist())


def load_idrid(data_dir: str | Path, size: int = 28) -> tuple[Split, Split]:
    """Load the IDRiD train and test splits from the layout described in the README."""
    data_dir = Path(data_dir)
    train = load_split(data_dir / "Excel" / "Training_Labels.csv", data_dir / "Training", size)
    test = load_split(data_dir / "Excel" / "Testing_Labels.csv", data_dir / "Testing", size)
    return train, test


def oversample(x: np.ndarray, y: np.ndarray, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Randomly repeat minority-class samples until every class matches the largest one.

    IDRiD is heavily imbalanced (grade 1 has 20 training images, grade 2 has 136),
    and an unweighted loss lets the classifier settle on predicting one grade.
    """
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    target = counts.max()
    picked = []
    for cls, count in zip(classes, counts):
        members = np.flatnonzero(y == cls)
        extra = rng.choice(members, size=target - count, replace=True)
        picked.append(np.concatenate([members, extra]))
    order = rng.permutation(np.concatenate(picked))
    return x[order], y[order]
