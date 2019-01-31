import os
import time
import itertools
import numpy as np
import tensorflow as tf

from models.yolo import net
from models.yolo.YOLOv3 import YOLOv3
from models.yolo.util import config as cfg
from models.yolo.dataset.DataSet import DataSet


def get_best_threshold(data_dir, batch_size, net_name, checkpoint_dir, ensamble, training):

    print('==> Get train data...')
    dataloader = DataSet(data_dir, batch_size, training)
    val_data_size = dataloader.nbr_val_nms
    print('==> Finished!')

    print('==> Create YOLOv3')
    inputs_x = tf.placeholder(tf.float32, [None, cfg.IMG_HEIGHT, cfg.IMG_WIDTH, 3])
    model = net.__dict__[net_name](inputs_x, training)
    yolo_v3 = YOLOv3(model, None, batch_size=1, is_training=training)
    print('==> Finished!')

    best_threshold = [0.0, 0.0]
    best_score = 0.0

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

        for score_threshold, iou_threshold in itertools.product(cfg.SCORE_THRESHOLD_LIST, cfg.IOU_THRESHOLD_LIST):
            print('score_threshold: %f, iou_threshold: %f, max_boxes: %d' % (score_threshold, iou_threshold, cfg.MAX_BOXES))

            train_nms_image = dataloader.train_nms_loader()

            avg_score = 0.0

            train_nms_list = None
            image = None
            result_boxes, result_score, result_classes = yolo_v3.predict(score_threshold, iou_threshold, cfg.MAX_BOXES, ensamble)

            while True:
                try:
                    train_nms_list = next(train_nms_image)
                    image = np.expand_dims(train_nms_list[0], axis=0)
                    ground_truth_box = train_nms_list[1]

                    box = sess.run(result_boxes, feed_dict={inputs_x : image})
                    
                    box_x2y2 = box[:, [3, 2]]
                    box_x1y1 = box[:, [1, 0]]
                    box = np.concatenate((box_x1y1, box_x2y2), axis=-1)

                    bucket = np.zeros(9)

                    for candidateBound in box:
                        
                        x1 = max(0, np.floor(candidateBound[0] + 0.5).astype(int))
                        y1 = max(0, np.floor(candidateBound[1] + 0.5).astype(int))
                        x2 = min(1069, np.floor(candidateBound[2] + 0.5).astype(int))
                        y2 = min(500, np.floor(candidateBound[3] + 0.5).astype(int))

                        best_iou = 0.0

                        for groundTruthBound in ground_truth_box:

                            iou = calculateIoU([x1, y1, x2, y2], groundTruthBound)

                            if iou > best_iou:
                                best_iou = iou

                        if best_iou >= 0.5:
                                bucket[0] += 1
                        if best_iou >= 0.55:
                            bucket[1] += 1
                        if best_iou >= 0.6:
                            bucket[2] += 1
                        if best_iou >= 0.65:
                            bucket[3] += 1
                        if best_iou >= 0.7:
                            bucket[4] += 1
                        if best_iou >= 0.75:
                            bucket[5] += 1
                        if best_iou >= 0.8:
                            bucket[6] += 1
                        if best_iou >= 0.85:
                            bucket[7] += 1
                        if best_iou >= 0.9:
                            bucket[8] += 1
                    
                    num_box = len(box)
                    num_eval_box = len(ground_truth_box)

                    each_image_score = 0.0
                    for n in bucket:
                        each_image_score += (2*n) / (num_box + num_eval_box)
                    each_image_score /= 9

                    avg_score += each_image_score

                except StopIteration:
                    break

            avg_score /= 1000
            if avg_score > best_score:
                best_score = avg_score
                best_threshold[0] = score_threshold
                best_threshold[1] = iou_threshold
            
            print('--- avg_score: %f' % avg_score)
            print('*** best score: %f' % best_score)
            print('*** best score_threshold: %f, best iou_threshold: %f' % (best_threshold[0], best_threshold[1]))
        


def calculateIoU(candidateBound, groundTruthBound):
    cx1 = candidateBound[0]
    cy1 = candidateBound[1]
    cx2 = candidateBound[2]
    cy2 = candidateBound[3]

    gx1 = groundTruthBound[0]
    gy1 = groundTruthBound[1]
    gx2 = groundTruthBound[2]
    gy2 = groundTruthBound[3]

    carea = (cx2 - cx1) * (cy2 - cy1)  # C的面积
    garea = (gx2 - gx1) * (gy2 - gy1)  # G的面积

    x1 = max(cx1, gx1)
    y1 = max(cy1, gy1)
    x2 = min(cx2, gx2)
    y2 = min(cy2, gy2)
    w = max(0, x2 - x1)
    h = max(0, y2 - y1)
    area = w * h  # C∩G的面积

    iou = area / (carea + garea - area)

    return iou