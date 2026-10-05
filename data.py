"""UTKFace loading. File names look like  [age]_[gender]_[race]_[date&time].jpg"""
from pathlib import Path

import numpy as np
import tensorflow as tf

from config import BATCH_SIZE, DATA_DIR, IMG_SIZE, MAX_AGE, SEED


def list_samples(data_dir=DATA_DIR):
    paths, ages, genders = [], [], []
    for p in sorted(Path(data_dir).glob("*.jpg")):
        parts = p.name.split("_")
        if len(parts) < 4:
            continue                      # a few UTKFace files have broken names
        try:
            age, gender = int(parts[0]), int(parts[1])
        except ValueError:
            continue
        if gender not in (0, 1) or not (1 <= age <= 116):
            continue
        paths.append(str(p))
        ages.append(age)
        genders.append(gender)
    if not paths:
        raise FileNotFoundError(f"No valid UTKFace images found in '{data_dir}'.")
    return (np.array(paths), np.array(ages, dtype="float32"),
            np.array(genders, dtype="float32"))


def split_indices(n):
    """Fixed 70 / 15 / 15 train / val / test split."""
    idx = np.random.default_rng(SEED).permutation(n)
    a, b = int(0.70 * n), int(0.85 * n)
    return idx[:a], idx[a:b], idx[b:]


def make_dataset(paths, ages, genders, preprocess_fn, training):
    ds = tf.data.Dataset.from_tensor_slices((paths, ages, genders))
    if training:
        ds = ds.shuffle(4096, seed=SEED)

    def load(path, age, gender):
        img = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
        img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))          # float32, 0-255
        if training:                                              # augmentation
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_brightness(img, 25.0)
            img = tf.clip_by_value(img, 0.0, 255.0)
        img = preprocess_fn(img)
        return img, {"gender": tf.reshape(gender, [1]),
                     "age": tf.reshape(age / MAX_AGE, [1])}

    return (ds.map(load, num_parallel_calls=tf.data.AUTOTUNE)
              .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE))
