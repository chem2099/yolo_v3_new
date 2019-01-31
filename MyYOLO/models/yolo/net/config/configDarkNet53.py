import numpy as np
from models.yolo.config import config as yolov3_cfg

NUM_CLASS = len(yolov3_cfg.CLASSES)
IMAGE_SIZE = 416
# CELL_SIZE = np.array([13, 26, 52])
# NUM_ANCHORS = 3
# ANCHORS = np.array([[45.068, 27.935],
#                     [76.403, 43.393],
#                     [88.214, 85.294],
#                     [124.864, 118.636],
#                     [136.438, 52.152],
#                     [163.376, 160.992],
#                     [209.207, 196.178],
#                     [227.513, 88.571],
#                     [294.034, 332.865]])
# ANCHOR_MASK = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

CELL_SIZE = np.array([13, 26, 52])
NUM_ANCHORS = 5
ANCHORS = np.array( [[ 42.  18.]
                     [ 50.  34.],
                     [ 69.  50.],
                     [ 84.  74.],
                     [ 84.  24.],
                     [108.  94.],
                     [125.  45.],
                     [131. 122.],
                     [139. 165.],
                     [167. 143.],
                     [168.  74.],
                     [178. 197.],
                     [179. 169.],
                     [217. 194.],
                     [302. 337.]])
ANCHOR_MASK = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14]]

DROP_RATE = 0.2

# leakyrelu alpha
ALPHA = 0.1

X_SCALE = 416 / 1069
Y_SCALE = 416 / 500
