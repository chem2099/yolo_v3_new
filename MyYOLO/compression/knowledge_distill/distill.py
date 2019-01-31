import tensorflow as tf
import os
import time
import math

from .Teacher import Teacher
from .Student import Student
from models.yolo.util import config as cfg
from models.yolo.dataset.DataSet import DataSet

def distill(student_net_name,
            data_dir, batch_size, net_name,
            epochs, start_epoch, start_iter,
            train_data_size, val_data_size,
            learning_rate, decay_rate, decay_steps,
            val_rate, save_rate,
            checkpoint_dir, log_dir,
            training):

    print('==> Get train and test data...')
    dataloader = DataSet(data_dir, batch_size, training)
    train_1w_batch = dataloader.train_1w_loader()
    print('==> Finished!')

    print('==> Create YOLOv3')
    print('--- use ', net_name)

    print('==> Finished!')

    print('==> Get each scale total loss')
    inputs_x = tf.placeholder(tf.float32, [None, cfg.IMG_HEIGHT, cfg.IMG_WIDTH, 3])

    total_grid_cell_attr = 5 + cfg.NUM_CLASS
    inputs_y = [tf.placeholder(tf.float32, [None, cfg.SCALES[0], cfg.SCALES[0], total_grid_cell_attr]),
                tf.placeholder(tf.float32, [None, cfg.SCALES[1], cfg.SCALES[1], total_grid_cell_attr]),
                tf.placeholder(tf.float32, [None, cfg.SCALES[2], cfg.SCALES[2], total_grid_cell_attr])]

    student = Student(inputs_x, student_net_name, batch_size)

    student_result_list = student.get_scale_list()

    scale0_mse_loss = tf.reduce_mean(tf.square(inputs_y[0] - student_result_list[0]))
    scale1_mse_loss = tf.reduce_mean(tf.square(inputs_y[1] - student_result_list[1]))
    scale2_mse_loss = tf.reduce_mean(tf.square(inputs_y[2] - student_result_list[2]))

    loss = scale0_mse_loss + scale1_mse_loss + scale2_mse_loss

    tf.summary.scalar('distill loss', loss)
    print('==> Finished!')

    print('==> Create optimizer')
    print('--- epochs = %d' % epochs)
    print('--- train_data_size = %d' % train_data_size)
    print('--- learning_rate = %f' % learning_rate)

    print('--- update learning_rate: ')
    print('--- \tlearning_rate = learning_rate * decay_rate^(global_step / decay_step)')
    print('--- decay_rate = %f' % decay_rate)

    total_step = epochs * math.ceil(train_data_size / batch_size)
    print('--- total_step = %d' % total_step)

    print('--- start_epochs = %d' % start_epoch)

    train_iter_max = math.ceil(train_data_size / batch_size)
    print('--- iter_max = %d' % train_iter_max)

    val_iter_max = math.ceil(val_data_size / batch_size)
    print('--- iter_max = %d' % val_iter_max)

    global_step = start_epoch * train_iter_max + start_iter
    print('--- global_step = %d' % global_step)
    global_step = tf.Variable(start_epoch * train_iter_max + start_iter, trainable=False)

    # learning_rate = learning_rate * decay_rate^(global_step / decay_steps)
    learning_rate = tf.train.exponential_decay(learning_rate, global_step,
                                               decay_steps, decay_rate, staircase=False)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss, global_step=global_step)
    print('==> Finished!')

    init_op = tf.group(tf.global_variables_initializer())

    saver = tf.train.Saver(tf.global_variables())

    summary_op = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(log_dir, flush_secs=60)

    with tf.Session() as sess:
        print('==> Load checkpoing')
        if len(os.listdir(checkpoint_dir)) >= 4:
            print('--> Restoring checkpoint from: ' + checkpoint_dir)

            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                sess.run(tf.local_variables_initializer())

            print('==> Load finished!')
        else:
            print('==> No checkpoint, start training new model')
            print('==> Init global variables')

            sess.run(init_op)
            train_writer.add_graph(sess.graph)

            print('==> Init finished!')

        print('==> Training start')

        epoch = start_epoch
        print('--- epoch = %d' % epoch)

        iter = start_iter
        print('--- iter = %d' % iter)

        summary_iter = 5

        print('--- save_rate = %d' % save_rate)

        print('--- val_rate = %d' % val_rate)

        save_path = os.path.join(checkpoint_dir)
        start_time = time.time()

        step = 0
        val_iter = 0
        total_val_loss = 0
        best_loss = 2147483647

        while epoch < epochs:
            while iter < train_iter_max:

                # ================== train ==================
                batch = next(train_1w_batch)

                feed_dict = {
                    inputs_x: batch[0],
                    # inputs_y[0]: batch[1][0],
                    # inputs_y[1]: batch[1][1],
                    # inputs_y[2]: batch[1][2]
                }

                total_loss, _, summary_str = sess.run([loss, train_op, summary_op], feed_dict=feed_dict)
                eta = remain_time(start_time, total_step, step)
                print('--- Epoch {}, Iter {}, ETA {:.2f}m, loss {:.3f}'.format(epoch, iter, eta, total_loss))
                # ================== train ==================

                # ================== val ==================
                if (step + 1) % val_rate == 0:

                    val_batch = dataloader.val_1w_loader()

                    for batch in val_batch:
                        feed_dict = {
                            inputs_x: batch[0],
                            inputs_y[0]: batch[1][0],
                            inputs_y[1]: batch[1][1],
                            inputs_y[2]: batch[1][2]
                        }
                        val_loss, summary_str = sess.run([loss, summary_op], feed_dict=feed_dict)
                        total_val_loss += val_loss

                    total_val_loss /= val_iter_max

                    print('-- Val loss {:.3f}'.format(total_val_loss), end=' ')

                    if total_val_loss < best_loss:
                        tmp = best_loss
                        best_loss = total_val_loss
                        print('Better then {:.3f}'.format(tmp))

                        model_name = save_path + os.sep + 'yolov3.ckpt' + '-epoch_' + str(epoch) + '_' + \
                                     str(iter) + '-bestloss_' + '%.3f' % best_loss
                        saver.save(sess, save_path=model_name, global_step=step)
                        print('--- save checkpoint best_loss: %.3f' % best_loss)
                    else:
                        print('Not better than {:.3f}'.format(best_loss))

                    total_val_loss = 0
                # ================== val ==================

                # write log
                if (step + 1) % summary_iter == 0:
                    train_writer.add_summary(summary_str, step)

                # save model
                if (step + 1) % save_rate == 0:
                    model_name = save_path + os.sep + 'yolov3.ckpt' + '-epoch_' + str(epoch) + '_' + \
                                 str(iter) + '-loss_' + '%.3f' % total_loss
                    saver.save(sess, save_path=model_name, global_step=step)
                    print('--- save checkpoint loss: %.3f' % total_loss)

                start_time = time.time()

                step += 1
                iter += 1
                val_iter += 1
                global_step += 1

            iter = 0
            epoch += 1

        print('==> Training Finished!')


def remain_time(start_time, total_step, step):
    end_time = time.time()
    during_time = end_time - start_time
    eta_s = (total_step - step) * during_time
    return eta_s / 60






