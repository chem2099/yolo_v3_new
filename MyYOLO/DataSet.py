import os
import cv2
import random
import numpy as np
import math
from models.yolo.util import config as cfg
# from models.yolo.config import config as cfg
from models.yolo.util.processing import preprocess_input

class DataSet:
    def __init__(self, data_dir, batch_size, training):
        batch_size =1
        self.batch_size = batch_size

        if training:
            self.train_1w_filename_path = os.path.join('data', data_dir, 'train', 'train_1w.csv')
            self.train_b_filename_path = os.path.join('data', data_dir, 'train', 'train_b.csv')
            self.train_nms_filename_path = os.path.join('data', data_dir, 'train', 'train_nms.csv')

            self.val_1w_filename_path = os.path.join('data', data_dir, 'val', 'val_1w.csv')
            self.val_b_filename_path = os.path.join('data', data_dir, 'val', 'val_b.csv')
        else:
            self.test_filename_path = os.path.join('data', data_dir, 'test', 'test_a.csv')

    def train_1w_loader(self):
        print('--- return train_1w data generator')

        data_list = open(self.train_1w_filename_path, 'r').readlines()
        batch_size = self.batch_size
        return_label = True
        img_width = cfg.IMG_WIDTH
        img_height = cfg.IMG_HEIGHT
        shuffle = False
        nbr_classes = cfg.NUM_CLASS
        grid_attr = cfg.GRID_ATTR

        return  generator_batch(data_list, grid_attr=grid_attr, nbr_classes=nbr_classes,
                                batch_size=batch_size, return_label=return_label, img_height=img_height,
                                img_width=img_width, shuffle=shuffle)

    def train_b_loader(self):
        print('--- return train_b data generator')

        data_list = open(self.train_b_filename_path, 'r').readlines()
        batch_size = self.batch_size
        return_label = True
        img_width = cfg.IMG_WIDTH
        img_height = cfg.IMG_HEIGHT
        shuffle = True
        nbr_classes = cfg.NUM_CLASS
        grid_attr = cfg.GRID_ATTR

        return  generator_batch(data_list, grid_attr=grid_attr, nbr_classes=nbr_classes,
                                batch_size=batch_size, return_label=return_label, img_height=img_height,
                                img_width=img_width, shuffle=shuffle)

    def val_1w_loader(self):
        print('--- return val_1w data generator')

        data_list = open(self.val_1w_filename_path, 'r').readlines()
        batch_size = self.batch_size
        img_width = cfg.IMG_WIDTH
        img_height = cfg.IMG_HEIGHT
        nbr_classes = cfg.NUM_CLASS
        grid_attr = cfg.GRID_ATTR

        return generator_val_batch(data_list, grid_attr=grid_attr, nbr_classes=nbr_classes,
                                   batch_size=batch_size, img_height=img_height, img_width=img_width)

    def val_b_loader(self):
        print('--- return val_b data generator')

        data_list = open(self.val_b_filename_path, 'r').readlines()
        batch_size = self.batch_size
        img_width = cfg.IMG_WIDTH
        img_height = cfg.IMG_HEIGHT
        nbr_classes = cfg.NUM_CLASS
        grid_attr = cfg.GRID_ATTR

        return  generator_val_batch(data_list, grid_attr=grid_attr, nbr_classes=nbr_classes,
                                    batch_size=batch_size, img_height=img_height, img_width=img_width)

    def train_nms_loader(self):
        print('--- return train nms data loader')

        batch_size = self.batch_size

        with open(self.train_nms_filename_path, 'r', encoding='UTF-8') as file:
            file.readline()
            for batch in file.readlines(batch_size):

                X_batch = np.zeros((len(batch), cfg.IMG_HEIGHT, cfg.IMG_WIDTH, 3))
                Y_batch = [[] for i in range(len(batch))]

                for i, line in enumerate(batch):
                    line = line.strip('\n')
                    label_list = line.split(',')
                    image_name = label_list[0]
                    position_list = list(map(lambda x : x.split('_'), label_list[1].split(';')))

                    image_path = os.path.join('data', 'car', 'image', 'val', image_name)
                    image = cv2.imread(image_path)

                    image = cv2.resize(image, (cfg.IMG_HEIGHT, cfg.IMG_WIDTH), interpolation=cv2.INTER_AREA)
                    image = preprocess_input(image)

                    X_batch[i] = image

                    boxes = []
                    for position in position_list:
                        if position[0] == '':
                            boxes += [0,0,0,0]

                        w = float(position[2])
                        h = float(position[3])
                        x1 = float(position[0])
                        y1 = float(position[1])
                        x2 = float(position[0]) + w
                        y2 = float(position[1]) + h
                        boxes += [x1, y1, x2, y2]

                    boxes = np.array(boxes).reshape(-1, 4)

                    Y_batch[i].append(boxes)

                yield (X_batch, Y_batch)

    def test_loader(self):
        print('--- return test data loader')

        with open(self.test_filename_path, 'r', encoding='UTF-8') as file:
            for image_path in file:
                image_path = image_path.strip('\n')

                image = cv2.imread(image_path)

                image = cv2.resize(image, (cfg.IMG_HEIGHT, cfg.IMG_WIDTH), interpolation=cv2.INTER_AREA)
                image = preprocess_input(image)

                yield [image, image_path]

def generator_batch(data_list, grid_attr, nbr_classes=2, batch_size=8,
                    return_label=True, img_width=416, img_height=416, shuffle=True):

    N = len(data_list)

    total_grid_attr = grid_attr + nbr_classes

    if shuffle:
        random.shuffle(data_list)

    batch_index = 0
    while True:
        current_index = (batch_index * batch_size) % N
        if N >= (current_index + batch_size):
            current_batch_size = batch_size
            batch_index += 1
        else:
            current_batch_size = N - current_index
            batch_index = 0
            if shuffle:
                random.shuffle(data_list)

        print('current_index: ', current_index)
        print('batch_index: ', batch_index)

        X_batch = np.zeros((current_batch_size, img_height, img_width, 3))
        Y_batch_scale0 = np.zeros((current_batch_size, cfg.SCALES[0], cfg.SCALES[0], total_grid_attr))
        Y_batch_scale1 = np.zeros((current_batch_size, cfg.SCALES[1], cfg.SCALES[1], total_grid_attr))
        Y_batch_scale2 = np.zeros((current_batch_size, cfg.SCALES[2], cfg.SCALES[2], total_grid_attr))

        # 把数据转换成网络需要的格式
        for i in range(current_index, current_index + current_batch_size):
            line = data_list[i].strip().split(',')

            img_path = line[0]
            print(img_path)

            position_list = list(map(lambda x : x.split('_'), line[1].split(';')))

            img = cv2.imread(img_path)

            # 对数据做一些处理
            img = cv2.resize(img, (img_height, img_width), interpolation=cv2.INTER_AREA)

            img = preprocess_input(img)

            scales_labels = []

            for scale_size in cfg.SCALES:
                maxtric = np.zeros((scale_size, scale_size, total_grid_attr), np.float32)

                for position in position_list:
                    if position[0] == '':
                        continue

                    w = float(position[2]) * cfg.X_SCALE
                    h = float(position[3]) * cfg.Y_SCALE
                    x = (float(position[0]) * cfg.X_SCALE + w / 2) / (416 / scale_size)
                    y = (float(position[1]) * cfg.Y_SCALE + h / 2) / (416 / scale_size)

                    grid_x = int(x)
                    grid_y = int(y)

                    maxtric[grid_y, grid_x, 0] = 1.0
                    maxtric[grid_y, grid_x, 1] = x
                    maxtric[grid_y, grid_x, 2] = y
                    maxtric[grid_y, grid_x, 3] = w
                    maxtric[grid_y, grid_x, 4] = h
                    maxtric[grid_y, grid_x, 5] = 1.0

                scales_labels.append(maxtric)

            X_batch[i - current_index] = img
            Y_batch_scale0[i - current_index] = scales_labels[0]
            Y_batch_scale1[i - current_index] = scales_labels[1]
            Y_batch_scale2[i - current_index] = scales_labels[2]

        X_batch = X_batch.astype(np.float32)
        Y_batch = [Y_batch_scale0, Y_batch_scale1, Y_batch_scale2]

        if return_label:
            yield (X_batch, Y_batch)
        else:
            yield X_batch

def generator_val_batch(data_list, grid_attr, nbr_classes=2, batch_size=8,
                        img_width=416, img_height=416):
    N = int(math.ceil(len(data_list) / batch_size))

    total_grid_attr = grid_attr + nbr_classes

    batch_index = 0
    for _ in range(N):
        current_index = (batch_index * batch_size) % N
        if N >= (current_index + batch_size):
            current_batch_size = batch_size
            batch_index += 1
        else:
            current_batch_size = N - current_index
            batch_index = 0

        X_batch = np.zeros((current_batch_size, img_height, img_width, 3))
        Y_batch_scale0 = np.zeros((current_batch_size, cfg.SCALES[0], cfg.SCALES[0], total_grid_attr))
        Y_batch_scale1 = np.zeros((current_batch_size, cfg.SCALES[1], cfg.SCALES[1], total_grid_attr))
        Y_batch_scale2 = np.zeros((current_batch_size, cfg.SCALES[2], cfg.SCALES[2], total_grid_attr))

        # 把数据转换成网络需要的格式
        for i in range(current_index, current_index + current_batch_size):
            line = data_list[i].strip().split(',')

            img_path = line[0]
            position_list = list(map(lambda x : x.split('_'), line[1].split(';')))

            img = cv2.imread(img_path)

            # 对数据做一些处理
            img = cv2.resize(img, (img_height, img_width), interpolation=cv2.INTER_AREA)

            img = preprocess_input(img)

            scales_labels = []

            for scale_size in cfg.SCALES:
                maxtric = np.zeros((scale_size, scale_size, total_grid_attr), np.float32)

                for position in position_list:
                    if position[0] == '':
                        continue

                    w = float(position[2]) * cfg.X_SCALE
                    h = float(position[3]) * cfg.Y_SCALE
                    x = (float(position[0]) * cfg.X_SCALE + w / 2) / (416 / scale_size)
                    y = (float(position[1]) * cfg.Y_SCALE + h / 2) / (416 / scale_size)

                    grid_x = int(x)
                    grid_y = int(y)

                    maxtric[grid_y, grid_x, 0] = 1.0
                    maxtric[grid_y, grid_x, 1] = x
                    maxtric[grid_y, grid_x, 2] = y
                    maxtric[grid_y, grid_x, 3] = w
                    maxtric[grid_y, grid_x, 4] = h
                    maxtric[grid_y, grid_x, 5] = 1.0

                scales_labels.append(maxtric)

            X_batch[i - current_index] = img
            Y_batch_scale0[i - current_index] = scales_labels[0]
            Y_batch_scale1[i - current_index] = scales_labels[1]
            Y_batch_scale2[i - current_index] = scales_labels[2]

        X_batch = X_batch.astype(np.float32)
        Y_batch = [Y_batch_scale0, Y_batch_scale1, Y_batch_scale2]

        yield (X_batch, Y_batch)




if __name__ == '__main__':

    import time
    import pandas as pd

    dataset = DataSet('car', 1, True)
    generator = dataset.train_1w_loader()
    # generator = dataset.train_b_loader(None)
    # generator = dataset.val_loader()

    x_zoom_rate = 416 / 1069
    y_zoom_rate = 416 / 500

    scales_size = [13, 26, 52]
    num_class = 1
    total_grid_attr = 5 + num_class

    each_tfrecord_size = 2000

    data_batch = None

    with open(os.path.join('data', 'car', 'train', 'train_1w.csv'), 'r', encoding='utf-8') as src_file:

        for line in src_file:

            for batch in generator:

                data_batch = batch

                print('label scale0')
                print(data_batch[1][0].shape)
                print(data_batch[1][0][data_batch[1][0] > 0.0])
                print('label scale1')
                print(data_batch[1][1].shape)
                print(data_batch[1][1][data_batch[1][1] > 0.0])
                print('label scale2')
                print(data_batch[1][2].shape)
                print(data_batch[1][2][data_batch[1][2] > 0.0])
                break

            line = line.strip('\n')
            label_list = line.split(',')
            image_name = label_list[0]
            position_list = list(map(lambda x: x.split('_'), label_list[1].split(';')))

            labels = []

            for scale_size in scales_size:
                maxtric = np.zeros((scale_size, scale_size, total_grid_attr), np.float32)

                for position in position_list:
                    if position[0] == '':
                        continue

                    w = float(position[2]) * x_zoom_rate
                    h = float(position[3]) * y_zoom_rate
                    x = (float(position[0]) * x_zoom_rate + w / 2) / (416 / scale_size)
                    y = (float(position[1]) * y_zoom_rate + h / 2) / (416 / scale_size)

                    grid_x = int(x)
                    grid_y = int(y)

                    maxtric[grid_y, grid_x, 0] = 1.0
                    maxtric[grid_y, grid_x, 1] = x
                    maxtric[grid_y, grid_x, 2] = y
                    maxtric[grid_y, grid_x, 3] = w
                    maxtric[grid_y, grid_x, 4] = h
                    maxtric[grid_y, grid_x, 5] = 1.0

                maxtric = np.expand_dims(maxtric, 0)
                labels.append(maxtric)

            print('test label scale0')
            print(labels[0].shape)
            print(labels[0][labels[0] > 0.0])
            print('test label scale1')
            print(labels[1].shape)
            print(labels[1][labels[1] > 0.0])
            print('test label scale2')
            print(labels[2].shape)
            print(labels[2][labels[2] > 0.0])


            time.sleep(20)


