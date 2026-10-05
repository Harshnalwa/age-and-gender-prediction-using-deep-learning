"""Face detection + ensemble inference, shared by the GUI and the image mode."""
from pathlib import Path

import cv2
import numpy as np
from tensorflow import keras

import models
from config import AGE_CALIBRATION, BACKBONES, GENDER_LABELS, IMG_SIZE, MAX_AGE, MODEL_DIR
from ensemble import ensemble


class AgeGenderPredictor:
    def __init__(self, model_dir=MODEL_DIR):
        self.nets = {}
        for name in BACKBONES:
            path = Path(model_dir) / f"{name}.keras"
            if path.exists():
                self.nets[name] = keras.models.load_model(path, compile=False)
        if not self.nets:
            raise FileNotFoundError(
                f"No trained models in '{model_dir}'. Run  python train.py  first.")
        print("Loaded models:", ", ".join(self.nets))
        self.detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def detect(self, frame):
        gray = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        return self.detector.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))

    def predict(self, frame):
        """Returns a list of dicts: box, gender, confidence, age."""
        boxes = self.detect(frame)
        if len(boxes) == 0:
            return []

        crops = []
        for x, y, w, h in boxes:
            m = int(0.2 * w)                                   # context margin
            y0, y1 = max(y - m, 0), min(y + h + m, frame.shape[0])
            x0, x1 = max(x - m, 0), min(x + w + m, frame.shape[1])
            rgb = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)
            crops.append(cv2.resize(rgb, (IMG_SIZE, IMG_SIZE)))
        batch = np.stack(crops).astype("float32")

        g_all, a_all = [], []
        for name, net in self.nets.items():
            x = models.BACKBONES[name][1](batch.copy())        # per-model preprocessing
            out = net(x, training=False)
            g_all.append(np.asarray(out["gender"]).ravel())
            a_all.append(np.asarray(out["age"]).ravel() * MAX_AGE)

        is_female, conf, age = ensemble(np.stack(g_all), np.stack(a_all))
        return [{"box": tuple(int(v) for v in b),
                 "gender": GENDER_LABELS[bool(f)],
                 "confidence": float(np.clip(c, 0.0, 1.0)),
                 "age": float(np.clip(a * AGE_CALIBRATION, 0.0, MAX_AGE))}
                for b, f, c, a in zip(boxes, is_female, conf, age)]


def draw(frame, results):
    for r in results:
        x, y, w, h = r["box"]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 0), 2)
        text = f'{r["gender"]}, ~{r["age"]:.0f} yrs ({r["confidence"] * 100:.0f}%)'
        cv2.putText(frame, text, (x, max(y - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
    return frame
