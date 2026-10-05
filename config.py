"""Central settings for the Age & Gender project."""
from pathlib import Path

IMG_SIZE = 128            # input resolution for all three backbones
BATCH_SIZE = 32
SEED = 42
MAX_AGE = 120.0           # ages are scaled to [0, 1] by dividing with this
AGE_CALIBRATION = 0.82    # mild on-device correction: UTKFace tends to over-estimate age
BACKBONES = ["vgg16", "resnet50v2", "xception"]

DATA_DIR = Path("data/UTKFace")      # put the UTKFace .jpg files here
MODEL_DIR = Path("saved_models")     # trained models are stored here

GENDER_LABELS = {False: "Male", True: "Female"}   # UTKFace: 0 = male, 1 = female
