import cv2
import numpy as np

# image_name = './data/car/image/train_1w/4aae5d41-c6bf-4d38-943a-fb3f7eea3692.jpg'
# boxes = [[607,4,91,62], [500,242,175,163], [498,0,92,36], [846,2,58,34]]
#
# def draw(image_name, boxes):
#     print('draw ' + image_name)
#
#     image = cv2.imread(image_name)
#
#     line = image_name.split('/')[-1] + ','
#     for box in boxes:
#         x, y, w, h = box
#         if w * h <= 120.0:
#             continue
#         if w * h >= 178167:
#             continue
#         x1 = max(0, np.floor(x + 0.5).astype(int))
#         y1 = max(0, np.floor(y + 0.5).astype(int))
#         x2 = min(image.shape[1], np.floor(x + w + 0.5).astype(int))
#         y2 = min(image.shape[0], np.floor(y + h + 0.5).astype(int))
#
#         cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
#
#     cv2.imwrite('./data/predict/' + image_name.split('/')[-1], image)
#
# draw(image_name, boxes)


Y_batch = [[] for i in range(8)]

print(Y_batch)
