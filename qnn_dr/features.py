"""Classical stage: turns each image into a handful of features, one per qubit.

Two extractors are available:
- "autoencoder": a small convolutional autoencoder trained on the IDRiD training
  images (the original design of the project).
- "pretrained": ImageNet ResNet50 features at 224x224, reduced with PCA.
"""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import MinMaxScaler


def build_autoencoder(img_size: int = 28, latent_dim: int = 8):
    """Return (autoencoder, encoder).

    The encoder maps an (img_size, img_size, 3) image to `latent_dim` values in
    [0, 1]; the decoder reconstructs all three colour channels.
    """
    from tensorflow import keras
    from tensorflow.keras import layers

    if img_size % 4:
        raise ValueError("img_size must be divisible by 4")
    small = img_size // 4

    image = keras.Input(shape=(img_size, img_size, 3), name="image")
    x = layers.Conv2D(16, 3, padding="same", activation="relu")(image)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Flatten()(x)
    latent = layers.Dense(latent_dim, activation="sigmoid", name="latent")(x)
    encoder = keras.Model(image, latent, name="encoder")

    code = keras.Input(shape=(latent_dim,), name="code")
    y = layers.Dense(small * small * 32, activation="relu")(code)
    y = layers.Reshape((small, small, 32))(y)
    y = layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu")(y)
    y = layers.Conv2DTranspose(16, 3, strides=2, padding="same", activation="relu")(y)
    reconstruction = layers.Conv2D(3, 3, padding="same", activation="sigmoid")(y)
    decoder = keras.Model(code, reconstruction, name="decoder")

    autoencoder = keras.Model(image, decoder(encoder(image)), name="autoencoder")
    autoencoder.compile(optimizer="adam", loss="mse")
    return autoencoder, encoder


def fit_encoder(
    train_images: np.ndarray,
    latent_dim: int = 8,
    epochs: int = 30,
    batch_size: int = 32,
    seed: int = 42,
    verbose: int = 2,
):
    """Train the autoencoder on training images only and return (encoder, history).

    A slice of the training set is held out for validation, so the test images
    are never seen before evaluation.
    """
    from tensorflow import keras

    keras.utils.set_random_seed(seed)
    autoencoder, encoder = build_autoencoder(train_images.shape[1], latent_dim)
    history = autoencoder.fit(
        train_images,
        train_images,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        shuffle=True,
        verbose=verbose,
    )
    return encoder, history


def encode(encoder, images: np.ndarray) -> np.ndarray:
    return np.asarray(encoder.predict(images, verbose=0), dtype=np.float64)


def pretrained_features(images: np.ndarray, batch_size: int = 16) -> np.ndarray:
    """2048-d ImageNet ResNet50 features (global average pooled) for each image.

    `images` are (N, H, W, 3) floats in [0, 1]; 224x224 matches ResNet50's
    training resolution. Weights are downloaded by Keras on first use.
    """
    from tensorflow import keras

    model = keras.applications.ResNet50(
        include_top=False, weights="imagenet", pooling="avg", input_shape=images.shape[1:]
    )
    preprocess = keras.applications.resnet50.preprocess_input
    out = []
    for start in range(0, len(images), batch_size):
        batch = preprocess(images[start : start + batch_size] * 255.0)
        out.append(model.predict(batch, verbose=0))
    return np.concatenate(out).astype(np.float64)


def reduce_to_qubits(train_features: np.ndarray, test_features: np.ndarray, n_components: int, seed: int = 42):
    """Standardise, then project onto the top `n_components` principal components
    of the training features (one component per qubit)."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(train_features)
    pca = PCA(n_components=n_components, random_state=seed).fit(scaler.transform(train_features))
    return pca.transform(scaler.transform(train_features)), pca.transform(scaler.transform(test_features))


def scale_to_angles(train_features: np.ndarray, test_features: np.ndarray):
    """Rescale features to [0, pi] for the ZFeatureMap rotation angles.

    The scaler is fitted on the training features only; test features outside the
    training range are clipped.
    """
    scaler = MinMaxScaler(feature_range=(0.0, np.pi)).fit(train_features)
    train_angles = np.clip(scaler.transform(train_features), 0.0, np.pi)
    test_angles = np.clip(scaler.transform(test_features), 0.0, np.pi)
    return train_angles, test_angles
