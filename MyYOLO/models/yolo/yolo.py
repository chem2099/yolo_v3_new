from models.yolo.train import train
from models.yolo.test import test
from models.yolo.util.grid_search_nms_threshold import get_best_threshold

__all__ = ['yolov3']

def yolov3(args):

    if args.training == 'True':

        train(args.data_dir, args.batch_size, args.net_name,
              args.epochs, args.start_epoch, args.start_iter, args.change_train_data_epoch,
              args.learning_rate, args.decay_rate, args.decay_steps,
              args.val_rate, args.save_rate,
              args.checkpoint_dir, args.log_dir,
              True)
        
    else:
        if args.select_threshold == 'True':
            get_best_threshold(args.data_dir, args.batch_size, args.net_name,
                                args.checkpoint_dir, args.ensamble, False)
        else:
            test(args.data_dir, args.net_name, args.ensamble, args.num_ensamble, args.num_ensamble_model, args.return_ensamble_result,
                args.score_threshold, args.iou_threshold, args.max_boxes,
                args.checkpoint_dir, args.result_path, False, args.draw_image)



