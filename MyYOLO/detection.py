from __future__ import print_function

import argparse
import os
import models

from utils import mkdir_p


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='TensorFlow Detection Training')
# Datasets
parser.add_argument('-dd', '--data_dir', default='car', type=str)
# Checkpoints
parser.add_argument('-ckptd', '--checkpoint_dir', default='checkpoint/YOLOv3/darknet53', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint), if you want to train a new model, please remove the checkpoint dir')
parser.add_argument('-sr', '--save_rate', default='1000', type=int, metavar='PATH',
                    help='save rate which will to save model(default: 500)')
# Log output dir
parser.add_argument('-ld', '--log_dir', default='logs', type=str, metavar='PATH',
                    help='training log dir, used for tensorboard')
# Architecture
parser.add_argument('-a', '--arch', default='yolov3', metavar='ARCH',
                    help='model atchitecture: \nif you want training model please use train_modelname\n' +
                    'if you want eval model please use eval_modelname')
parser.add_argument('-nn', '--net_name', default='darknet53', metavar='NET',
                    help='net of the arch')
# Train or Predict
parser.add_argument('-t', '--training', default='True', type=str, metavar='N',
                    help='predict model on test set')
# Train
parser.add_argument('-vr', '--val_rate', default=50, type=int, metavar='N',
                    help='val rate')
parser.add_argument('-e', '--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-se', '--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-si', '--start_iter', default=0, type=int, metavar='N',
                    help='manual iter number (useful on restarts)')
parser.add_argument('-ctde', '--change_train_data_epoch', default=2, type=int, metavar='N',
                    help='change_train_data_epoch')
parser.add_argument('-bs', '--batch_size', default=2, type=int, metavar='N',
                    help='batch_size')
# Test
parser.add_argument('-rp', '--result_path', default='./data/predict/', type=str, metavar='N',
                    help='result_path')
parser.add_argument('-di', '--draw_image', default=False, type=bool, metavar='N',
                    help='draw image when testing')
parser.add_argument('-selt', '--select_threshold', default='True', type=str, metavar='N',
                    help='select_threshold')
parser.add_argument('-ens', '--ensamble', default='False', type=str, metavar='N',
                    help='is ensamble')
parser.add_argument('-nens', '--num_ensamble', default='0', type=str, metavar='N',
                    help='num ensamble')
parser.add_argument('-rer', '--return_ensamble_result', default='False', type=str, metavar='N',
                    help='num return_ensamble_result')
parser.add_argument('-nem', '--num_ensamble_model', default=5, type=int, metavar='N',
                    help='num_ensamble_model')
# Optimization options
parser.add_argument('-lr', '--learning_rate', default=0.01, type=float, metavar='N',
                    help='the learning rate of optimizator')
parser.add_argument('-dr', '--decay_rate', default=0.7, type=float, metavar='N',
                    help='the decay rate of learning rate')
parser.add_argument('-ds', '--decay_steps', default=2000, type=int, metavar='N',
                    help='the decay step of learning rate')
# NMS config
parser.add_argument('-mb', '--max_boxes', default=35, type=int, metavar='N',
                    help='the max boxes of nms')
parser.add_argument('-it', '--iou_threshold', default=0.5, type=float, metavar='N',
                    help='the nms iou threshold')
parser.add_argument('-st', '--score_threshold', default=0.3, type=float, metavar='N',
                    help='the yolo score threshold')
# Device options
#parser.add_argument('--gpu-id', default='', type=str,
                    #help='id(s) for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Validate dataset
assert args.data_dir == 'car', 'Dataset can only be car.'

# Select GPU Device
#os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

def main():
    if args.training == 'True':
        if not os.path.isdir(args.checkpoint_dir):
            mkdir_p(args.checkpoint_dir)
        print("==> training model '{}'".format(args.arch))
    else:
        print("==> testing model '{}'".format(args.arch))
    models.__dict__[args.arch](args)


if __name__ == '__main__':
    main()

