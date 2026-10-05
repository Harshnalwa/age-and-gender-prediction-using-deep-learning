"""Evaluate each model and the voting ensemble on the held-out test split."""
import numpy as np
from tensorflow import keras

import data
import models
from config import BACKBONES, MAX_AGE, MODEL_DIR
from ensemble import ensemble


def main():
    paths, ages, genders = data.list_samples()
    _, _, te = data.split_indices(len(paths))
    y_age, y_gender = ages[te], genders[te] == 1

    g_all, a_all = [], []
    for name in BACKBONES:
        path = MODEL_DIR / f"{name}.keras"
        if not path.exists():
            print(f"skipping {name} (no saved model)")
            continue
        model = keras.models.load_model(path, compile=False)
        ds = data.make_dataset(paths[te], ages[te], genders[te],
                               models.BACKBONES[name][1], False)
        out = model.predict(ds, verbose=0)
        g, a = out["gender"].ravel(), out["age"].ravel() * MAX_AGE
        g_all.append(g)
        a_all.append(a)
        print(f"{name:12s} gender acc {np.mean((g >= 0.5) == y_gender):.4f} | "
              f"age MAE {np.mean(np.abs(a - y_age)):.2f} yrs")

    if len(g_all) > 1:
        is_female, _, age = ensemble(np.stack(g_all), np.stack(a_all))
        print(f"{'ENSEMBLE':12s} gender acc {np.mean(is_female == y_gender):.4f} | "
              f"age MAE {np.mean(np.abs(age - y_age)):.2f} yrs")


if __name__ == "__main__":
    main()
