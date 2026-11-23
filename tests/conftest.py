"""Builds a tiny fake IDRiD folder so the pipeline can be tested without the real data."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _fake_fundus(path: Path, grade: int, rng: np.random.Generator) -> None:
    """A coloured disc on a wide black canvas, brighter for higher grades."""
    img = Image.new("RGB", (120, 80), "black")
    draw = ImageDraw.Draw(img)
    base = 60 + 35 * grade
    colour = tuple(int(np.clip(base + rng.integers(-15, 15), 0, 255)) for _ in range(3))
    draw.ellipse((25, 5, 95, 75), fill=colour)
    img.save(path, "JPEG")


def _write_split(root: Path, folder: str, csv_name: str, per_class: int, rng) -> None:
    (root / folder).mkdir(parents=True, exist_ok=True)
    rows = []
    n = 0
    for grade in range(5):
        for _ in range(per_class):
            n += 1
            name = f"IDRiD_{n:03d}"
            _fake_fundus(root / folder / f"{name}.jpg", grade, rng)
            rows.append({"Image name": name, "Retinopathy grade": grade, "Risk of macular edema ": 0})
    df = pd.DataFrame(rows)
    df["Unnamed: 3"] = np.nan  # the real CSVs carry empty trailing columns
    df.to_csv(root / "Excel" / csv_name, index=False)


@pytest.fixture()
def fake_idrid(tmp_path: Path) -> Path:
    root = tmp_path / "IDRiD"
    (root / "Excel").mkdir(parents=True)
    rng = np.random.default_rng(0)
    _write_split(root, "Training", "Training_Labels.csv", per_class=4, rng=rng)
    _write_split(root, "Testing", "Testing_Labels.csv", per_class=2, rng=rng)
    return root
