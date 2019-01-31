import numpy as np
import os
from imgaug import augmenters as iaa
import cv2

# with open('./data/car/image/test_a/test_a.csv', 'w') as file:
#     # img_list = os.listdir('I:\DC大赛\交通卡口车辆信息精准识别\project\data\src_data\\test_a')
#     img_list = os.listdir('./data/car/image/test_a/')
#
#     for img_name in img_list:
#         file.write(img_name + '\n')

# seq = iaa.Sequential(
#     [
#         iaa.GaussianBlur(0.1)
#     ]
# )
# image_path = './data/car/image/val/3333bf6f-57cb-4d90-afe1-636dcd88a255.jpg'
# target_image_path = './1.jpg'
# image = cv2.imread(image_path)
# image = seq.augment_images(image)
# cv2.imwrite(target_image_path, image)
#
#
# seq = iaa.Sequential(
#     [
#         iaa.Sharpen(0.07)
#     ]
# )

# import cv2
# from models.yolo.util.deHaze import deHaze
# from imgaug import augmenters as iaa


# image_paths = ['./data/car/image/test/1.jpg', './data/car/image/test/2.jpg', './data/car/image/test/3.jpg',
#               './data/car/image/test/4.jpg', './data/car/image/test/5.jpg', './data/car/image/test/6.jpg',
#                './data/car/image/test/7.jpg', './data/car/image/test/8.jpg']


# def gamma_trans(img, gamma):  # gamma函数处理
#     gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]  # 建立映射表
#     gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)  # 颜色值为整数
#     return cv2.LUT(img, gamma_table)  # 图片颜色查表。另外可以根据光强（颜色）均匀化原则设计自适应算法


# seq = iaa.Sequential(
#         [
#             iaa.Sharpen(0.1)
#         ]
# )

# kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)

# def preprocess_input(img):
#     # img = cv2.resize(img, (416, 416), interpolation=cv2.INTER_AREA)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     print(img.shape)
#     img = cv2.bilateralFilter(img, 0, 10, 5)
#     #img = seq.augment_images(img)
#     #img = cv2.filter2D(img, -1, kernel=kernel)
#     #img = gamma_trans(img, 1.3)

#     #img = img / 255
#     #img = deHaze(img) * 255

#     return img

# for i, image_path in enumerate(image_paths):
#     target_image_path = './data/car/image/test/' + 'test' + str(i) + '.jpg'
#     image = cv2.imread(image_path)

#     image = preprocess_input(image)

#     cv2.imwrite(target_image_path, image)


with open('./data/car/train/train_nms_1.csv', 'w') as targ:
    with open('./data/car/train/train_nms.csv', 'r') as src:
        src.readline()
        for line in src:
            targ.write('data/car/image/train_b/' + line)