# Age & Gender Prediction - Real-Time Deep Learning System

Project implementation of the research paper **"Age and Gender Prediction Using Deep Learning"**
(Chandigarh University). A webcam application that detects faces and predicts **gender** and
**age** using an ensemble of three CNNs (VGG16, ResNet50V2, Xception) trained on **UTKFace**.

## How it works

```
Webcam frame -> Haar face detection -> crop + resize (128x128)
             -> VGG16 / ResNet50V2 / Xception (each: gender head + age head)
             -> voting ensemble (majority vote for gender, mean for age)
             -> label overlay in Tkinter window
```

| File | Purpose |
|---|---|
| `config.py` | image size, paths, backbone list |
| `data.py` | UTKFace parsing, 70/15/15 split, tf.data pipeline + augmentation |
| `models.py` | the three transfer-learning models, fine-tuning helper |
| `train.py` | two-phase training (heads -> fine-tune) |
| `evaluate.py` | per-model and ensemble accuracy / MAE on the test split |
| `ensemble.py` | voting logic |
| `predictor.py` | face detection + ensemble inference |
| `app.py` | Tkinter webcam GUI (and `--image` mode) |

## Setup

Use Python 3.9-3.12. From PowerShell:

```bash
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

On macOS or Linux, activate the environment with `source .venv/bin/activate`.

Download **UTKFace** (the "aligned & cropped" images) from https://susanqq.github.io/UTKFace/
or Kaggle, and put all `.jpg` files in `data/UTKFace/`.

## Train the models

The trained model files are not included in this repository. Training creates
`saved_models/vgg16.keras`, `saved_models/resnet50v2.keras`, and
`saved_models/xception.keras`; the webcam app and evaluation script expect these files.
Training all three models can take substantial time and is best run with a supported GPU.

## Run

```bash
python train.py --backbone all     # trains all 3 (GPU strongly recommended, e.g. Colab)
python evaluate.py                 # accuracy / MAE for each model + ensemble
python app.py                      # live webcam GUI
python app.py --image photo.jpg    # single image
```

Tip: on a slow laptop use `python app.py --every 5` (models run every 5th frame).

## Paper's reported results (to compare with yours)

Gender accuracy **96.8%**, age MAE **+/-4.3 years**, 25-30 FPS with GPU. Your numbers
will depend on training time and hardware - report what `evaluate.py` prints.

## Notes / limitations

- Haar cascades work best on frontal, well-lit faces; swap in MTCNN or a DNN detector for harder scenes.
- Age/gender estimation from faces is biased by dataset and can be wrong - treat outputs as estimates,
  and get consent before using it on other people's faces.
