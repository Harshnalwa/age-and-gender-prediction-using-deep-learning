"""Real-time webcam GUI (Tkinter).

  python app.py                    # live webcam
  python app.py --image face.jpg   # single image -> face_out.jpg
"""
import argparse
import time
import tkinter as tk
from tkinter import messagebox
from pathlib import Path

import cv2
from PIL import Image, ImageTk

from predictor import AgeGenderPredictor, draw


class App:
    def __init__(self, root, predictor, camera, every):
        self.root, self.predictor, self.every = root, predictor, every
        self.cap = cv2.VideoCapture(camera)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open the webcam.")
        self.results, self.n, self.last, self.frame = [], 0, time.time(), None
        self.read_failures = 0

        root.title("Age & Gender Prediction - VGG16 + ResNet50V2 + Xception")
        self.video = tk.Label(root)
        self.video.pack()
        self.status = tk.Label(root, text="", anchor="w")
        self.status.pack(fill="x")
        bar = tk.Frame(root)
        bar.pack(pady=4)
        tk.Button(bar, text="Save snapshot", command=self.snapshot).pack(side="left", padx=6)
        tk.Button(bar, text="Quit", command=self.close).pack(side="left", padx=6)
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.update()

    def update(self):
        ok, frame = self.cap.read()
        if not ok:
            self.read_failures += 1
            if self.read_failures >= 10:
                messagebox.showerror(
                    "Camera unavailable",
                    "The webcam is not providing video frames. Check that it is connected "
                    "and that camera access is enabled.",
                    parent=self.root,
                )
                self.close()
                return
            self.root.after(100, self.update)
            return

        self.read_failures = 0
        frame = cv2.flip(frame, 1)
        if self.n % self.every == 0:           # predict every N-th frame for speed
            self.results = self.predictor.predict(frame)
        self.n += 1
        draw(frame, self.results)
        self.frame = frame.copy()

        now = time.time()
        fps = 1.0 / max(now - self.last, 1e-6)
        self.last = now
        self.status.config(text=f"  FPS: {fps:4.1f}   Faces: {len(self.results)}")

        img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        self.video.configure(image=img)
        self.video.image = img
        self.root.after(10, self.update)

    def snapshot(self):
        if self.frame is not None:
            name = f"snapshot_{int(time.time())}.jpg"
            cv2.imwrite(name, self.frame)
            self.status.config(text=f"  Saved {name}")

    def close(self):
        self.cap.release()
        self.root.destroy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--every", type=int, default=3, help="run the models every N frames")
    ap.add_argument("--image", help="predict on one image instead of the webcam")
    args = ap.parse_args()

    predictor = AgeGenderPredictor()
    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            raise SystemExit(f"Cannot read image: {args.image}")
        results = predictor.predict(frame)
        for r in results:
            print(f'{r["gender"]}, age ~{r["age"]:.0f}, confidence {r["confidence"]:.2f}')
        if not results:
            print("No face found.")
        out = Path(args.image).with_name(Path(args.image).stem + "_out.jpg")
        cv2.imwrite(str(out), draw(frame, results))
        print("Saved", out)
    else:
        root = tk.Tk()
        App(root, predictor, args.camera, args.every)
        root.mainloop()


if __name__ == "__main__":
    main()
