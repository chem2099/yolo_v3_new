import numpy as np
import tensorflow as tf
from models.yolo.config import config as cfg

class YOLOv3(object):
    def __init__(self, net, labels, batch_size, is_training=True):
        # yolov3 config
        self.num_class = len(cfg.CLASSES)
        self.batch_size = batch_size
        self.ignore_thresh = cfg.IGNORE_THRESH

        # net config
        self.image_size = net.image_size
        self.cell_size = net.cell_size
        self.scale = self.image_size / self.cell_size
        self.num_anchors = net.num_anchors
        self.anchors = net.anchors
        self.anchor_mask = net.anchor_mask
        self.x_scale = net.x_scale
        self.y_scale = net.y_scale

        # loss config
        self.object_alpha = cfg.OBJECT_ALPHA
        self.no_object_alpha = cfg.NO_OBJECT_ALPHA
        self.class_alpha = cfg.CLASS_ALPHA
        self.coord_alpha = cfg.COORD_ALPHA
        self.attention_alpha = cfg.ATTENTION_ALPHA

        # total loss
        self.loss = 0

        if is_training:
            self.scales, self.attention = net.get_output()
            self.total_loss(labels)
        else:
            self.scales = net.get_output()

    def calculate_IOU(self, pred_box, label_box, scope='IOU'):
        with tf.name_scope(scope):
            with tf.name_scope('pred_box'):
                pred_box_xy = pred_box[..., 0:2]
                pred_box_wh = pred_box[..., 2:]
                pred_box_wh_half = pred_box_wh / 2.0
                pred_box_leftup = pred_box_xy - pred_box_wh_half
                pred_box_rightdown = pred_box_xy + pred_box_wh_half
                pred_box_area = pred_box_wh[..., 0] * pred_box_wh[..., 1]

            with tf.name_scope('label_box'):
                label_box_xy = label_box[..., 0:2]
                label_box_wh = label_box[..., 2:]
                label_box_wh_half = label_box_wh / 2.0
                label_box_leftup = label_box_xy - label_box_wh_half
                label_box_rightdown = label_box_xy + label_box_wh_half
                label_box_area = label_box_wh[..., 0] * label_box_wh[..., 1]

            with tf.name_scope('intersection'):
                intersection_leftup = tf.maximum(pred_box_leftup, label_box_leftup)
                intersection_rightdown = tf.minimum(pred_box_rightdown, label_box_rightdown)
                intersection_wh = tf.maximum(intersection_rightdown - intersection_leftup, 0.)
                intersection_area = intersection_wh[..., 0] * intersection_wh[..., 1]

            iou = tf.divide(intersection_area, (pred_box_area + label_box_area - intersection_area), name='IOU_result')

        return iou

    def calculate_object_confidence_loss(self, object_confidence_hat, object_confidence, object_mask, best_confidence_mask, scope='object_confidence_loss'):
        with tf.name_scope(scope):
            object_loss = tf.reduce_sum(object_mask * tf.square(object_confidence - object_confidence_hat) * best_confidence_mask) / self.batch_size

        return object_loss

    def calculate_no_object_confidence_loss(self, no_object_confidence_hat, no_object_confidence, object_mask, ignore_mask, scope='no_object_confidence_loss'):
        with tf.name_scope(scope):
            no_object_loss = tf.reduce_sum(object_mask * tf.square(no_object_confidence - no_object_confidence_hat) * ignore_mask) / self.batch_size

        return no_object_loss

    def calculate_xy_loss(self, label_object_mask, best_confidence_mask, txy_hat, txy, scope='xy_loss'):
        with tf.name_scope(scope):
            xy_loss = tf.reduce_sum(label_object_mask * best_confidence_mask * tf.square(txy - txy_hat)) / self.batch_size

        return xy_loss

    def calculate_wh_loss(self, label_object_mask, best_confidence_mask, twh_hat, twh, scope='wh_loss'):
        with tf.name_scope(scope):
            wh_loss = tf.reduce_sum(label_object_mask * best_confidence_mask * tf.square(twh - twh_hat)) / self.batch_size

        return wh_loss

    def calculate_classify_loss(self, object_mask, predicts_class, labels_class, scope='classify_loss'):
        with tf.name_scope(scope):
            class_loss = tf.reduce_sum(object_mask * tf.keras.backend.binary_crossentropy(labels_class, predicts_class, from_logits=True)) / self.batch_size
            # class_loss = tf.reduce_mean(tf.keras.backend.binary_crossentropy(labels_class, predicts_class, from_logits=True), axis=0)

        return class_loss

    def calculate_attention_loss(self, attention_scale, label_attention_scale, scope='attention_loss'):
        with tf.name_scope(scope):
            attention_loss = tf.reduce_mean(tf.keras.backend.binary_crossentropy(label_attention_scale, attention_scale, from_logits=True)) / self.batch_size

        return attention_loss

    def scale_hat(self, feature_map, anchors, calculate_loss=False):
        anchors_tensor = tf.cast(anchors, tf.float32)
        anchors_tensor = tf.reshape(anchors_tensor, [1, 1, 1, self.num_anchors, 2])

        with tf.name_scope('create_grid_offset'):
            grid_shape = tf.shape(feature_map)[1:3]
            grid_y = tf.range(0, grid_shape[0])
            grid_x = tf.range(0, grid_shape[1])
            grid_y = tf.reshape(grid_y, [-1, 1, 1, 1]) # (13, 1, 1, 1)
            grid_x = tf.reshape(grid_x, [1, -1, 1, 1]) # (1, 13, 1, 1)
            grid_y = tf.tile(grid_y, [1, grid_shape[0], 1, 1]) # (13, 13, 1, 1)
            grid_x = tf.tile(grid_x, [grid_shape[1], 1, 1, 1]) # (13, 13, 1, 1)
            grid = tf.concat([grid_x, grid_y], axis=3)
            grid = tf.cast(grid, tf.float32)

        feature_map = tf.reshape(feature_map, [-1, grid_shape[0], grid_shape[1], self.num_anchors, 5 + self.num_class])
        
        with tf.name_scope('scale_hat_activations'):
            tf.summary.histogram('feature_map', feature_map)

            bbox_confidence = tf.sigmoid(feature_map[..., 0:1], name='confidence')
            tf.summary.histogram('confidence', bbox_confidence)
            bbox_xy = tf.sigmoid(feature_map[..., 1:3], name='xy')
            tf.summary.histogram('xy', bbox_xy)
            bbox_wh = tf.exp(feature_map[..., 3:5], name='wh')
            tf.summary.histogram('wh', bbox_wh)
            bbox_class_probs = tf.sigmoid(feature_map[..., 5:], name='class_probs')
            tf.summary.histogram('class_probs', bbox_class_probs)

            bbox_xy = (bbox_xy + grid) / tf.cast(grid_shape[0], tf.float32)
            bbox_wh = bbox_wh * anchors_tensor / tf.cast(self.image_size, tf.float32)

        if calculate_loss:
            return grid, feature_map, bbox_xy, bbox_wh

        return bbox_xy, bbox_wh, bbox_confidence, bbox_class_probs

    def total_loss(self, labels, labels_attention, scope='total_loss'):
        with tf.name_scope(scope):
            grid_shape = [tf.cast(tf.shape(self.scales[i])[1:3], dtype=tf.float32) for i in range(len(self.scales))]

            for i in range(len(self.scales)):
                grid, predict, bbox_xy, bbox_wh = self.scale_hat(self.scales[i],
                                                                 self.anchors[self.anchor_mask[i]],
                                                                 calculate_loss=True)
                pred_bbox = tf.concat([bbox_xy, bbox_wh], axis=-1) # (None, 13, 13, 3, 4)

                # label has object grid
                label_object_mask = tf.cast(labels[i][..., 0:1], tf.float32)
                label_object_mask = tf.expand_dims(label_object_mask, axis=3)
                label_object_mask = tf.tile(label_object_mask, [1, 1, 1, self.num_anchors, 1])
                # label no object grid
                label_no_object_mask = 1.0 - label_object_mask
                # label xy
                label_xy = labels[i][..., 1:3]
                label_xy = tf.expand_dims(label_xy, axis=3) / tf.cast(grid_shape[i], tf.float32)
                label_xy = tf.tile(label_xy, [1, 1, 1, self.num_anchors, 1])
                # label wh
                label_wh = labels[i][..., 3:5] / tf.cast(self.image_size, tf.float32)
                label_wh = tf.expand_dims(label_wh, axis=3)
                label_wh = tf.tile(label_wh, [1, 1, 1, self.num_anchors, 1])
                # label class
                label_class = tf.cast(labels[i][..., 5:], tf.float32)
                label_class = tf.expand_dims(label_class, axis=3)
                label_class = tf.tile(label_class, [1, 1, 1, self.num_anchors, 1])


                label_box = tf.concat([label_xy, label_wh], axis=-1)

                # pred confidence
                confidence_hat = tf.sigmoid(predict[..., 0:1]) # (None, 13, 13, 3, 1)
                # pred xywh
                txy_hat = tf.sigmoid(predict[..., 1:3])
                #txy_hat = predict[..., 1:3]
                tf.summary.histogram('txy_hat', txy_hat)
                twh_hat = predict[..., 3:5]
                tf.summary.histogram('twh_hat', twh_hat)
                # pred class
                class_hat = predict[..., 5:]

                # ground truth xywh
                anchors_tensor = tf.cast(self.anchors[self.anchor_mask[i]], tf.float32)
                anchors_tensor = tf.reshape(anchors_tensor, [1, 1, self.num_anchors, 2])
                # twh = tf.log(labels[i][..., 3:5] / anchors_tensor)
                twh = tf.log(label_wh * tf.cast(self.image_size, tf.float32) / anchors_tensor)
                # avoid log(0)
                twh = tf.keras.backend.switch(label_object_mask, twh, tf.zeros_like(twh))
                txy = (label_xy * tf.cast(grid_shape[i], tf.float32) - grid) * label_object_mask

                # iou
                iou = self.calculate_IOU(pred_bbox, label_box) # (None, 13, 13, 3)
                # get best confidence for each grid cell
                best_confidence = tf.reduce_max(confidence_hat, axis=-1, keepdims=True) # (None, 13, 13, 1)
                best_confidence_mask = tf.cast(confidence_hat >= best_confidence, tf.float32) # (None, 13, 13, 3)

                # get ignore mask, if some anchor box of the object grid cell hasn't best iou but they may be has better iou with ground truth
                ignore_mask = tf.cast(iou < self.ignore_thresh, tf.float32) # (None, 13, 13, 3)
                ignore_mask = tf.expand_dims(ignore_mask, axis=4) # (None, 13, 13, 3, 1)

                label_object_confidence = 1
                label_no_object_confidence = 0

                # if not the best iou box, calculate no_object_mask
                no_object_mask = (1 - best_confidence_mask) + label_no_object_mask

                object_confidence_loss = self.calculate_object_confidence_loss(confidence_hat,
                                                                                label_object_confidence,
                                                                                label_object_mask,
                                                                                best_confidence_mask,
                                                                                'object_confidence_loss_' + str(i))
                no_object_confidence_loss = self.calculate_no_object_confidence_loss(confidence_hat, 
                                                                                     label_no_object_confidence, 
                                                                                     no_object_mask, 
                                                                                     ignore_mask, 
                                                                                     'no_object_confidence_loss_' + str(i))
                xy_loss = self.calculate_xy_loss(label_object_mask, best_confidence_mask, txy_hat, txy, 'xy_loss_' + str(i))
                wh_loss = self.calculate_wh_loss(label_object_mask, best_confidence_mask, twh_hat, twh, 'wh_loss_' + str(i))
                class_loss = self.calculate_classify_loss(label_object_mask, class_hat, label_class, 'classify_loss_' + str(i))
                attention_loss = self.calculate_attention_loss(self.attention[i], labels_attention[i])

                self.loss += self.coord_alpha * (xy_loss + wh_loss) + \
                             self.object_alpha * object_confidence_loss + \
                             self.no_object_alpha * no_object_confidence_loss + \
                             self.class_alpha * class_loss + \
                             self.attention_alpha * attention_loss

                tmp = tf.expand_dims(iou, axis=4)
                avg_iou = tf.reduce_mean(tf.boolean_mask(tmp, tf.cast(label_object_mask, tf.bool)))

                tf.summary.scalar('avg_iou' + str(i), avg_iou)
                tf.summary.scalar('object_confidence_loss' + str(i), object_confidence_loss)
                tf.summary.scalar('no_object_confidence_loss' + str(i), no_object_confidence_loss)
                tf.summary.scalar('xy_loss' + str(i), xy_loss)
                tf.summary.scalar('wh_loss' + str(i), wh_loss)
                tf.summary.scalar('class_loss' + str(i), class_loss)
                tf.summary.scalar('attention_loss' + str(i), attention_loss)
            tf.summary.scalar('loss', self.loss)

    def get_box_and_score(self, scale, anchors):
        bbox_xy, bbox_wh, bbox_confidence, bbox_class_probs = self.scale_hat(scale, anchors)

        score = bbox_confidence * bbox_class_probs
        score = tf.reshape(score, [-1, self.num_class])

        scale_size = scale.get_shape().as_list()[1]
        bbox_xy = bbox_xy * scale_size * (self.image_size / scale_size) / tf.cast([self.x_scale, self.y_scale], tf.float32)
        bbox_xy = tf.reshape(bbox_xy, [-1, 2])
        bbox_yx = bbox_xy[..., ::-1]
        
        bbox_wh = bbox_wh * self.image_size / tf.cast([self.x_scale, self.y_scale], tf.float32)
        bbox_wh = tf.reshape(bbox_wh, [-1, 2])
        bbox_hw = bbox_wh[..., ::-1]

        bbox_y1x1 = bbox_yx - (bbox_hw / 2.0)
        bbox_y2x2 = bbox_yx + (bbox_hw / 2.0)

        box = tf.concat([bbox_y1x1, bbox_y2x2], axis=-1)

        return  box, score

    def predict(self, score_threshold, iou_threshold, max_boxes, ensamble):
        boxes = []
        boxes_score = []

        for i in range(len(self.scales)):
            with tf.name_scope('predict' + str(i)):
                box, score = self.get_box_and_score(self.scales[i], self.anchors[self.anchor_mask[i]])

                boxes.append(box)
                boxes_score.append(score)

        boxes = tf.concat(boxes, axis=0)
        
        boxes_score = tf.concat(boxes_score, axis=0)

        if ensamble == 'True':
            return boxes, boxes_score

        mask = boxes_score >= score_threshold
        max_boxes = tf.constant(max_boxes, dtype='int32', name='max_boxes')

        result_boxes = []
        result_score = []
        result_classes = []
        for num in range(self.num_class):
            class_boxes = tf.boolean_mask(boxes, mask[:, num])

            class_boxes_score = tf.boolean_mask(boxes_score[:, num], mask[:, num])

            nms_index = tf.image.non_max_suppression(class_boxes,
                                                     class_boxes_score,
                                                     max_boxes,
                                                     iou_threshold=iou_threshold,
                                                     name='NMS')
            class_boxes = tf.gather(class_boxes, nms_index)
            class_boxes_score = tf.gather(class_boxes_score, nms_index)
            classes = tf.ones_like(class_boxes_score, 'int32')

            result_boxes.append(class_boxes)
            result_score.append(class_boxes_score)
            result_classes.append(classes)

        result_boxes = tf.concat(result_boxes, axis=0)
        result_score = tf.concat(result_score, axis=0)
        result_classes = tf.concat(result_classes, axis=0)

        return result_boxes, result_score, result_classes


if __name__ == '__main__':    
    import cv2
    data = cv2.imread('../../data/car/image/train_1w/a6fb5706-db1b-4ea7-920f-dfd0657660ee.jpg')
    data = cv2.resize(data, (416, 416))
    # data = data / 255
    data = tf.image.per_image_standardization(data)
    # data = tf.cast(tf.expand_dims(tf.constant(data), 0), tf.float32)
    data = tf.cast(tf.expand_dims(data, 0), tf.float32)

    x_zoom_rate = 416 / 1069
    y_zoom_rate = 416 / 500

    scales_size = [13, 26, 52]
    num_class = 1
    total_grid_attr = 5 + num_class

    line = 'a6fcaaac-75fe-487e-acc6-dde7b23a952a.jpg,98_0_51_35;226_0_43_26;407_0_96_14;12_0_102_69;169_0_80_62;661_1_104_56;604_134_138_158;890_188_160_144;465_415_231_85;938_430_128_69;174_235_187_178'
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

            print('x, y, w, h: %f, %f, %f, %f' % (x,y,w,h))

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
    # from models.yolo.net.DarkNet53 import DarkNet53
    # darknet53 = DarkNet53(data, True)

    from models.yolo.net.DenseNet import DenseNet
    densenet = DenseNet(data, True)
    yolo = YOLOv3(densenet, labels, 1)

    loss = yolo.loss

    args = {'max_boxes': 20, 'iou_threshold': 0.5, 'score_threshold': 0.45}

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # result_boxes, result_score, result_classes = yolo.predict(args)

        # print('result_boxes:')
        # print(sess.run(result_boxes))
        # print('result_score:')
        # print(sess.run(result_score))
        # print('result_classes:')
        # print(sess.run(result_classes))
        print(sess.run(loss))

    print()





