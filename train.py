"""Train the three backbones.   python train.py --backbone all"""
import argparse

from tensorflow import keras

import data
import models
from config import BACKBONES, MODEL_DIR


def compile_model(model, lr):
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss={"gender": "binary_crossentropy", "age": "mae"},
        loss_weights={"gender": 1.0, "age": 5.0},   # age loss is on a 0-1 scale
        metrics={"gender": ["accuracy"], "age": ["mae"]},
    )


def callbacks():
    return [keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.3, patience=2)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="all", choices=BACKBONES + ["all"])
    ap.add_argument("--epochs-head", type=int, default=5)
    ap.add_argument("--epochs-finetune", type=int, default=15)
    args = ap.parse_args()
    names = BACKBONES if args.backbone == "all" else [args.backbone]

    paths, ages, genders = data.list_samples()
    tr, va, _ = data.split_indices(len(paths))
    print(f"{len(paths)} images  |  train {len(tr)}  val {len(va)}")
    MODEL_DIR.mkdir(exist_ok=True)

    for name in names:
        pre = models.BACKBONES[name][1]
        train_ds = data.make_dataset(paths[tr], ages[tr], genders[tr], pre, True)
        val_ds = data.make_dataset(paths[va], ages[va], genders[va], pre, False)

        print(f"\n=== {name}: phase 1 (train heads) ===")
        model = models.build_model(name)
        compile_model(model, 1e-3)
        model.fit(train_ds, validation_data=val_ds,
                  epochs=args.epochs_head, callbacks=callbacks())

        print(f"\n=== {name}: phase 2 (fine-tune) ===")
        models.unfreeze(model)
        compile_model(model, 1e-5)
        model.fit(train_ds, validation_data=val_ds,
                  epochs=args.epochs_finetune, callbacks=callbacks())

        model.save(MODEL_DIR / f"{name}.keras")
        print(f"saved -> {MODEL_DIR / (name + '.keras')}")


if __name__ == "__main__":
    main()
