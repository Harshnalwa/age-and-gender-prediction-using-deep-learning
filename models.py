"""Backbone definitions: VGG16, ResNet50V2 and Xception with a shared two-head design."""
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import VGG16, ResNet50V2, Xception
from tensorflow.keras.applications import vgg16, resnet_v2, xception

from config import IMG_SIZE

# name -> (constructor, matching preprocess_input)
BACKBONES = {
    "vgg16": (VGG16, vgg16.preprocess_input),
    "resnet50v2": (ResNet50V2, resnet_v2.preprocess_input),
    "xception": (Xception, xception.preprocess_input),
}


def build_model(name: str) -> keras.Model:
    """Pretrained backbone (frozen) + shared dense layer + gender head + age head."""
    ctor, _ = BACKBONES[name]
    base = ctor(include_top=False, weights="imagenet",
                input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base.trainable = False

    inp = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    x = base(inp, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    gender = layers.Dense(1, activation="sigmoid", name="gender")(x)  # P(female)
    age = layers.Dense(1, activation="sigmoid", name="age")(x)        # age / MAX_AGE
    return keras.Model(inp, {"gender": gender, "age": age}, name=f"{name}_age_gender")


def unfreeze(model: keras.Model, last_n: int = 30) -> None:
    """Fine-tuning phase: unfreeze the last `last_n` backbone layers (BatchNorm stays frozen)."""
    base = next(l for l in model.layers if isinstance(l, keras.Model))
    base.trainable = True
    for layer in base.layers[:-last_n]:
        layer.trainable = False
    for layer in base.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
