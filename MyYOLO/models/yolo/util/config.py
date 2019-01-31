import numpy as np

SCALES = [13, 26, 52]
#SCALES = [26, 52, 104]
NUM_CLASS = 1
GRID_ATTR = 5

IMG_WIDTH = 416
IMG_HEIGHT = 416

X_SCALE = 416 / 1069
Y_SCALE = 416 / 500

SCORE_THRESHOLD_LIST = np.linspace(0.7, 0.8, 3, dtype=np.float32)
IOU_THRESHOLD_LIST = np.linspace(0.3, 0.4, 3, dtype=np.float32)
MAX_BOXES = 40

train_val_rate = 0.8