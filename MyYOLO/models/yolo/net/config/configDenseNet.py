import numpy as np
from models.yolo.config import config as yolov3_cfg

NUM_CLASS = len(yolov3_cfg.CLASSES)
IMAGE_SIZE = 416
CELL_SIZE = np.array([13, 26, 52])
NUM_ANCHORS = 5
ANCHORS = np.array([
    [ 39.07201754,  25.80140428],
    [ 65.80069799,  31.56669918],
    [ 66.88678829, 146.12497191],
    [ 70.89768814,  62.98424832],
    [ 99.83233678,  86.10561809],
    [110.42969977,  38.94441885],
    [128.08780003, 116.24260609],
    [153.98412516,  61.44485038],
    [159.62295651, 149.98699965],
    [172.29549498, 189.13133234],
    [221.17919093, 263.46911562],
    [214.65606772, 175.01036746],
    [223.47169047,  81.53708667],
    [318.79080674, 152.33352026],
    [323.17118454, 365.64912624]])
ANCHOR_MASK = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14]]

ALPHA = 0.1
GROTH_RATE = 48
DROP_RATE = 0.1
THETA = 4.0

# net config
#NET_CONFIG = [['D', 6], ['T', 1], ['D', 12], ['S', 2], ['T', 1], ['D', 12], ['S', 1], ['T', 1], ['D', 24], ['S', 0]]
NET_CONFIG = [['D', 4], ['T', 1], ['D', 4], ['S', 2], ['T', 1], ['D', 5], ['S', 1], ['T', 1], ['D', 5], ['S', 0]]


X_SCALE = 416 / 1069
Y_SCALE = 416 / 500