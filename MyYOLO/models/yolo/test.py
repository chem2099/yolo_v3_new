import os
import cv2
import time
import shutil
import glob
import numpy as np
import pandas as pd
import tensorflow as tf

from models.yolo import net
from models.yolo.YOLOv3 import YOLOv3
from models.yolo.util import config as cfg
from models.yolo.dataset.DataSet import DataSet


def test(data_dir, net_name, ensamble, num_ensamble, num_ensamble_model, return_ensamble_result,
         score_threshold, iou_threshold, max_boxes,
         checkpoint_dir, result_path, training, draw_image):

    if ensamble == 'False' and os.path.isdir(result_path):
        shutil.rmtree(result_path)
        os.mkdir(result_path)

    if return_ensamble_result == 'True':
        get_ensamble_result(result_path, data_dir, num_ensamble_model, score_threshold, iou_threshold, max_boxes, draw_image)
    else:
        print('==> Get train and test data...')
        dataloader = DataSet(data_dir, 1, training)
        test_image = dataloader.test_loader()
        print('==> Finished!')

        print('==> Create YOLOv3')
        inputs_x = tf.placeholder(tf.float32, [None, cfg.IMG_HEIGHT, cfg.IMG_WIDTH, 3])
        model = net.__dict__[net_name](inputs_x, training)
        yolo_v3 = YOLOv3(model, None, batch_size=1, is_training=training)
        print('==> Finished!')

        if ensamble == 'True':
            boxes, boxes_score = yolo_v3.predict(score_threshold, iou_threshold, max_boxes, ensamble)
        else:
            result_boxes, result_score, result_classes = yolo_v3.predict(score_threshold, iou_threshold, max_boxes, ensamble)

        restorer = tf.train.Saver()
        with tf.Session() as sess:

            print('==> Load checkpoing')
            if len(os.listdir(checkpoint_dir)) >= 4:
                print('--> Restoring checkpoint from: ' + checkpoint_dir)
                ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    restorer.restore(sess, ckpt.model_checkpoint_path)
                print('==> Load finished!')
            else:
                print('==> ckpt not exist or lack of required files')
                os._exit(-1)

            total_time = 0.0
            num = 1
            while True:
                try:
                    test_list = next(test_image)
                    image = np.expand_dims(test_list[0], axis=0)

                    start_time = time.time()
                    if ensamble == 'True':
                        box, score = sess.run([boxes, boxes_score],
                                            feed_dict={inputs_x : image})
                        
                        box_y2x2 = box[:, [2, 3]]
                        box_y1x1 = box[:, [0, 1]]
                        box = np.concatenate((box_y1x1, box_y2x2), axis=-1)

                        save_box_and_score(box, score, test_list[1], result_path, num_ensamble, num)
                        
                    else:
                        box, score, classes = sess.run([result_boxes, result_score, result_classes],
                                                feed_dict={inputs_x : image})

                        total_time += (time.time() - start_time)

                        box_wh = box[:, [3, 2]] - box[:, [1, 0]]
                        box_xy = box[:, [1, 0]]
                        box = np.concatenate((box_xy, box_wh), axis=-1)

                        draw(result_path, test_list[1], box, draw_image)

                    num += 1

                except StopIteration:
                    break
            
            if ensamble == 'False':
                print('fps: %.3f' % (float(num) / total_time))

def draw(result_path, image_name, boxes, draw_image):
    result_csv_path = os.path.join(result_path + 'result.csv')
    with open(result_csv_path, 'a+') as file:
        image = cv2.imread(image_name)

        line = image_name.split(os.sep)[-1] + ','

        for box in boxes:
            x, y, w, h = box

            if w * h <= 200.0:
                continue

            x1 = max(0, np.floor(x + 0.5).astype(int))
            y1 = max(0, np.floor(y + 0.5).astype(int))
            x2 = min(image.shape[1], np.floor(x + w + 0.5).astype(int))
            y2 = min(image.shape[0], np.floor(y + h + 0.5).astype(int))

            line += '{0}_{1}_{2}_{3};'.format(x1, y1, x2 - x1, y2 - y1)

            if draw_image:
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

        file.write(line[:-1] + '\n')
        if draw_image:
            image_path = os.path.join(result_path, image_name.split(os.sep)[-1])
            cv2.imwrite(image_path, image)

def save_box_and_score(boxs, scores, image_path, result_path, num_ensamble, num):

    num_dir = os.path.join(result_path, num_ensamble)

    if not os.path.isdir(num_dir):
        os.makedirs(num_dir)

    file_path = os.path.join(result_path, num_ensamble, image_path.split(os.sep)[-1].split('.')[0] + '.txt')

    print(num)

    with open(file_path, 'w') as file:
        for box, score in zip(boxs, scores):
            line = '{:.5f},{:.5f},{:.5f},{:.5f},{:.5f}'.format(box[0], box[1], box[2], box[3], score[0])
            file.write(line + '\n')

def get_ensamble_result(result_path, data_dir, num_ensamble_model, score_threshold, iou_threshold, max_boxes, draw_image):

    dataloader = DataSet(data_dir, 1, False)

    ensamble_file_list = []

    for i in range(num_ensamble_model):
        tmp = glob.glob(os.path.join(result_path, str(i), '*.txt'))
        ensamble_file_list.append({path.strip('.txt').split(os.sep)[-1] : path for path in tmp})


    # ========================= graph calculate =======================

    tf_boxes_score = tf.placeholder(tf.float32, [None, 1])
    tf_boxes = tf.placeholder(tf.float32, [None, 4])

    mask = tf_boxes_score >= score_threshold
    max_boxes = tf.constant(max_boxes, dtype='int32', name='max_boxes')

    result_boxes = []
    for num in range(1):
        class_boxes = tf.boolean_mask(tf_boxes, mask[:, num])

        class_boxes_score = tf.boolean_mask(tf_boxes_score[:, num], mask[:, num])

        nms_index = tf.image.non_max_suppression(class_boxes,
                                                class_boxes_score,
                                                max_boxes,
                                                iou_threshold=iou_threshold,
                                                name='NMS')
        class_boxes = tf.gather(class_boxes, nms_index)

        result_boxes.append(class_boxes)

    result_boxes = tf.concat(result_boxes, axis=0)

    # ========================= graph calculate =======================

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        with open(dataloader.test_filename_path, 'r') as test_file:
            for image_path in test_file:
                image_path = image_path.strip('\n')
                image_name = image_path.split(os.sep)[-1].split('.')[0]

                ensamble_file_path = ensamble_file_list[0][image_name]
                boxes_and_scores = pd.read_csv(ensamble_file_path, sep=',', header=None).values
                for i in range(1, num_ensamble_model):
                    ensamble_file_path = ensamble_file_list[i][image_name]
                    boxes_and_scores += pd.read_csv(ensamble_file_path, sep=',', header=None).values

                boxes_and_scores /= num_ensamble_model

                scores = boxes_and_scores[:, -1:]
                boxes = boxes_and_scores[:, 0:4]

                result = sess.run(result_boxes, feed_dict={tf_boxes_score : scores, tf_boxes : boxes})

                box_wh = result[:, [3, 2]] - result[:, [1, 0]]
                box_xy = result[:, [1, 0]]
                box = np.concatenate((box_xy, box_wh), axis=-1)

                draw(result_path, image_path, box, draw_image)



            




