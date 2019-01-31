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
        #batch_size =1
        self.batch_size = batch_size
        self.nbr_train_1w = 0
        self.nbr_train_b = 0
        self.nbr_val_1w = 0
        self.nbr_val_b = 0
        self.nbr_val_nms = 0
        #train_date.csv train_final_1990.csv  train_b.csv 
        if training:
            self.train_1w_filename_path = os.path.join('data', data_dir, 'train', 'train_date.csv')
            self.train_b_filename_path = os.path.join('data', data_dir, 'train', 'train_b.csv')

            self.val_1w_filename_path = os.path.join('data', data_dir, 'val', 'val_1w.csv')
            self.val_b_filename_path = os.path.join('data', data_dir, 'val', 'val_b.csv')
        else:
            self.train_nms_filename_path = os.path.join('data', data_dir, 'train', 'train_nms.csv')
            self.test_filename_path = os.path.join('data', data_dir, 'test', 'test_a.csv')

    def train_1w_loader(self):
        print('--- return train_1w data generator')

        data_list = open(self.train_1w_filename_path, 'r').readlines()
        self.nbr_train_1w = len(data_list)
        batch_size = self.batch_size
        return_label = True
        img_width = cfg.IMG_WIDTH
        img_height = cfg.IMG_HEIGHT
        shuffle = True
        nbr_classes = cfg.NUM_CLASS
        grid_attr = cfg.GRID_ATTR

        return  generator_batch(data_list, dataset_path='train_1w',grid_attr=grid_attr, nbr_classes=nbr_classes,
                                batch_size=batch_size, return_label=return_label, img_height=img_height,
                                img_width=img_width, shuffle=shuffle)

    def train_b_loader(self):
        print('--- return train_b data generator')

        data_list = open(self.train_b_filename_path, 'r').readlines()
        self.nbr_train_b = len(data_list)
        batch_size = self.batch_size
        return_label = True
        img_width = cfg.IMG_WIDTH
        img_height = cfg.IMG_HEIGHT
        shuffle = True
        nbr_classes = cfg.NUM_CLASS
        grid_attr = cfg.GRID_ATTR

        return  generator_batch(data_list, dataset_path='train_b',grid_attr=grid_attr, nbr_classes=nbr_classes,
                                batch_size=batch_size, return_label=return_label, img_height=img_height,
                                img_width=img_width, shuffle=shuffle)

    def val_1w_loader(self):
        print('--- return val_1w data generator')

        data_list = open(self.val_1w_filename_path, 'r').readlines()
        self.nbr_val_1w = len(data_list)
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
        self.nbr_val_b = len(data_list)
        batch_size = self.batch_size
        img_width = cfg.IMG_WIDTH
        img_height = cfg.IMG_HEIGHT
        nbr_classes = cfg.NUM_CLASS
        grid_attr = cfg.GRID_ATTR

        return  generator_val_batch(data_list, grid_attr=grid_attr, nbr_classes=nbr_classes,
                                    batch_size=batch_size, img_height=img_height, img_width=img_width)

    def train_nms_loader(self):
        print('--- return train nms data loader')

        self.nbr_val_nms = 1000
                    
        with open(self.train_nms_filename_path, 'r', encoding='UTF-8') as file:
            for line in file:
                line = line.strip('\n').strip(';')
                label_list = line.split(',')
                image_path = label_list[0]
                # print(image_path)
                position_list = list(map(lambda x : x.split('_'), label_list[1].split(';')))

                image = cv2.imread(image_path)
                image = cv2.resize(image, (cfg.IMG_HEIGHT, cfg.IMG_WIDTH), interpolation=cv2.INTER_AREA)
                image = preprocess_input(image)
                boxes = []
                # print(position_list)
                for position in position_list:
                    if position[0] == '':
                        boxes += [0,0,0,0]
                    
                        # print(position)
                    else:
                        
                        w = float(position[2])
                        h = float(position[3])
                        x1 = float(position[0])
                        y1 = float(position[1])
                        x2 = float(position[0]) + w
                        y2 = float(position[1]) + h
                        boxes += [x1, y1, x2, y2]

                boxes = np.array(boxes).reshape(-1, 4)

                yield [image, boxes]

    def test_loader(self):
        print('--- return test data loader')

        with open(self.test_filename_path, 'r', encoding='UTF-8') as file:
            for image_path in file:
                image_path = image_path.strip('\n')

                image = cv2.imread(image_path)

                image = cv2.resize(image, (cfg.IMG_HEIGHT, cfg.IMG_WIDTH), interpolation=cv2.INTER_AREA)
                image = preprocess_input(image)

                yield [image, image_path]

def generator_batch(data_list, dataset_path,grid_attr, nbr_classes=2, batch_size=1,
                    return_label=True, img_width=416, img_height=416, shuffle=True):
    #条件表达式 写成扁平化
    dataline=len(data_list)
    #if dataline >= 2:
        #N = dataline-1
    #else:
        #N = dataline
    N = dataline-1 if dataline >= 2 else dataline    
    new_data_list = []
    new_data_list_append = new_data_list.append
    for item in data_list:
        path_name , tag = item.strip().split(',')
        img_path = os.path.join(os.getcwd(), "data", "car", "image", dataset_path, path_name)
        if  os.path.isfile(img_path):
            new_data_list_append(f'{path_name},{tag}' )
            # continue
    data_list = new_data_list
    total_grid_attr = grid_attr + nbr_classes

    if shuffle:
        random.shuffle(data_list)

    batch_index = 0
    while True:
        current_index = (batch_index * batch_size) % N
        #index =min(N ,current_index + batch_size)
        if N >= (current_index + batch_size):
            current_batch_size = batch_size
            batch_index += 1
        else:
            current_batch_size = N - current_index
            batch_index = 0
            if shuffle:
                random.shuffle(data_list)

        X_batch = np.zeros((current_batch_size, img_width, img_height, 3))
        Y_batch_scale0 = np.zeros((current_batch_size, cfg.SCALES[0], cfg.SCALES[0], total_grid_attr))
        Y_batch_scale1 = np.zeros((current_batch_size, cfg.SCALES[1], cfg.SCALES[1], total_grid_attr))
        Y_batch_scale2 = np.zeros((current_batch_size, cfg.SCALES[2], cfg.SCALES[2], total_grid_attr))

        # 加载图片及匹配标签文件名()
        for i in range(current_index, current_index + current_batch_size):
            line = data_list[i].strip().strip(';').split(',')

            img_path = line[0]
            position_list = list(map(lambda x : x.split('_'), line[1].split(';')))
            # path='data/car/image/'+path+'/'
            path_img= os.path.join(os.getcwd(), "data", "car", "image", dataset_path, img_path)
            if not os.path.isfile(path_img):
                continue
            img = cv2.imread(path_img)
            img = cv2.resize(img, (cfg.IMG_HEIGHT, cfg.IMG_WIDTH), interpolation=cv2.INTER_AREA)

            #img = preprocess_input(img)

            scales_labels = []

            for scale_size in cfg.SCALES:
                maxtric = np.zeros((scale_size, scale_size, total_grid_attr), np.float32)

                for position in position_list:
                    if position[0] == '' or len(position) != 5:
                        continue

                    w = float(position[2]) * cfg.X_SCALE
                    h = float(position[3]) * cfg.Y_SCALE
                    x = (float(position[0]) * cfg.X_SCALE + w / 2) / (416 / scale_size)
                    y = (float(position[1]) * cfg.Y_SCALE + h / 2) / (416 / scale_size)
                    #c = float(position[4])  
                    c = 0
                    
                    grid_x = int(x)
                    grid_y = int(y)

                    maxtric[grid_y, grid_x, 0] = 1.0
                    maxtric[grid_y, grid_x, 1] = x
                    maxtric[grid_y, grid_x, 2] = y
                    maxtric[grid_y, grid_x, 3] = w
                    maxtric[grid_y, grid_x, 4] = h
                    maxtric[grid_y, grid_x, 5] = c

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

def generator_val_batch(data_list, grid_attr, nbr_classes=2, batch_size=1,
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

        X_batch = np.zeros((current_batch_size, img_width, img_height, 3))
        Y_batch_scale0 = np.zeros((current_batch_size, cfg.SCALES[0], cfg.SCALES[0], total_grid_attr))
        Y_batch_scale1 = np.zeros((current_batch_size, cfg.SCALES[1], cfg.SCALES[1], total_grid_attr))
        Y_batch_scale2 = np.zeros((current_batch_size, cfg.SCALES[2], cfg.SCALES[2], total_grid_attr))

        # 鎶婃暟鎹�浆鎹㈡垚缃戠粶闇��鐨勬牸寮�
        for i in range(current_index, current_index + current_batch_size):
            line = data_list[i].strip().split(',')

            img_path = line[0]
            position_list = list(map(lambda x : x.split('_'), line[1].split(';')))

            img = cv2.imread(img_path)

            # 瀵规暟鎹�仛涓�簺澶勭悊
            img = cv2.resize(img, (cfg.IMG_HEIGHT, cfg.IMG_WIDTH), interpolation=cv2.INTER_AREA)

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
                    c = 1 if len(position) < 5 else float(position[4])

                    grid_x = int(x)
                    grid_y = int(y)

                    maxtric[grid_y, grid_x, 0] = 1.0
                    maxtric[grid_y, grid_x, 1] = x
                    maxtric[grid_y, grid_x, 2] = y
                    maxtric[grid_y, grid_x, 3] = w
                    maxtric[grid_y, grid_x, 4] = h
                    maxtric[grid_y, grid_x, 5] = c

                scales_labels.append(maxtric)

            X_batch[i - current_index] = img
            Y_batch_scale0[i - current_index] = scales_labels[0]
            Y_batch_scale1[i - current_index] = scales_labels[1]
            Y_batch_scale2[i - current_index] = scales_labels[2]

        X_batch = X_batch.astype(np.float32)
        Y_batch = [Y_batch_scale0, Y_batch_scale1, Y_batch_scale2]


        yield (X_batch, Y_batch)




if __name__ == '__main__':

    dataset = DataSet('car', 8, True)
    generator = dataset.train_1w_loader()
    # generator = dataset.train_b_loader(None)
    # generator = dataset.val_loader()

    import time

    for batch in generator:

        for img in batch[0]:
            cv2.imwrite('./1.jpg', img)

            time.sleep(5)