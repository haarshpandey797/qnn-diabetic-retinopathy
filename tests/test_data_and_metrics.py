import numpy as np
from PIL import Image

from qnn_dr.data import crop_to_fundus, load_idrid, oversample, pad_to_square, read_labels
from qnn_dr.evaluate import baselines, format_table, metrics


def test_crop_removes_black_border():
    img = Image.new("RGB", (100, 60), "black")
    img.paste((200, 100, 50), (30, 10, 70, 50))
    assert crop_to_fundus(img).size == (40, 40)


def test_crop_ignores_stray_bright_pixels():
    img = Image.new("RGB", (100, 60), "black")
    img.paste((200, 100, 50), (30, 10, 70, 50))
    img.putpixel((2, 30), (255, 255, 255))  # JPEG noise near the edge
    assert crop_to_fundus(img).size == (40, 40)


def test_crop_keeps_all_black_image():
    img = Image.new("RGB", (10, 10), "black")
    assert crop_to_fundus(img).size == (10, 10)


def test_pad_to_square_centres_image():
    img = Image.new("RGB", (40, 20), (200, 0, 0))
    out = pad_to_square(img)
    assert out.size == (40, 40)
    assert out.getpixel((20, 5)) == (0, 0, 0)
    assert out.getpixel((20, 20)) == (200, 0, 0)


def test_read_labels_handles_messy_headers(fake_idrid):
    df = read_labels(fake_idrid / "Excel" / "Training_Labels.csv")
    assert list(df.columns) == ["Image name", "Retinopathy grade"]
    assert sorted(df["Retinopathy grade"].unique()) == [0, 1, 2, 3, 4]


def test_load_idrid_shapes_and_range(fake_idrid):
    train, test = load_idrid(fake_idrid, size=28)
    assert train.images.shape == (20, 28, 28, 3)
    assert test.images.shape == (10, 28, 28, 3)
    assert train.images.dtype == np.float32
    assert 0.0 <= train.images.min() and train.images.max() <= 1.0
    assert train.labels.tolist() == sorted(train.labels.tolist())


def test_oversample_balances_classes():
    x = np.arange(10).reshape(-1, 1)
    y = np.array([0] * 6 + [1] * 3 + [2] * 1)
    xb, yb = oversample(x, y, seed=0)
    assert np.bincount(yb).tolist() == [6, 6, 6]
    # every row still pairs a sample with its own label
    assert all(y[int(row[0])] == label for row, label in zip(xb, yb))


def test_majority_prediction_scores_like_class_share():
    y_true = np.array([2] * 136 + [0] * 134 + [1] * 20 + [3] * 74 + [4] * 49)
    m = metrics(y_true, np.full_like(y_true, 2))
    assert round(m["accuracy"], 4) == 0.3293  # what the 2023 notebook reported
    assert m["balanced_accuracy"] == 0.2


def test_baselines_run(fake_idrid):
    rng = np.random.default_rng(0)
    y = np.repeat(np.arange(5), 4)
    x = rng.uniform(0, np.pi, size=(20, 8)) + y[:, None] * 0.3
    res = baselines(x, y, x, y, y)
    assert set(res) == {"majority_class", "logistic_regression", "mlp"}
    assert "acc" in format_table(res)
