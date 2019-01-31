import cv2
from models.yolo.util.deHaze import deHaze
#from imgaug import augmenters as iaa

#seq = iaa.Sequential(
    #[
        #iaa.Sharpen(0.1)
    #]
#)

def preprocess_input(img):
    # img = seq.augment_image(img)
    img = cv2.bilateralFilter(img, 0, 10, 5)
    img = img / 255
    #img = deHaze(img)

    return img
