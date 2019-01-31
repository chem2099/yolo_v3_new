import tensorflow as tf
from models.yolo.net.config import configDenseNet as cfg

__all__ = ['densenet']

class DenseNet:
    def __init__(self, inputs, training):
        # net config
        self.num_class = cfg.NUM_CLASS
        self.image_size = cfg.IMAGE_SIZE
        self.cell_size = cfg.CELL_SIZE
        self.num_anchors = cfg.NUM_ANCHORS
        self.anchors = cfg.ANCHORS
        self.anchor_mask = cfg.ANCHOR_MASK
        self.x_scale = cfg.X_SCALE
        self.y_scale = cfg.Y_SCALE
        self.alpha = cfg.ALPHA
        self.growth_rate = cfg.GROTH_RATE
        self.drop_rate = cfg.DROP_RATE
        self.theta = cfg.THETA
        self.net_config = cfg.NET_CONFIG

        self.scale0 = None
        self.scale1 = None
        self.scale2 = None

        self._feature_extractor_inputs(inputs, training)
        self._detection_layer(training)

    def _feature_extractor_inputs(self, inputs, training=True):
        with tf.variable_scope('feature_extractor'):
            # inputs = self._Conv2d(inputs, 2 * self.growth_rate, kernal_size=[7, 7], stride=2, name='first_conv2d')
            # inputs = self._MaxPool2d(inputs, pool_size=[3, 3], strides=2)

            num_layer = 0

            # 0
            inputs = self._Conv2d(inputs, filters=32, kernal_size=[3, 3], stride=(1, 1), name='_Conv2d_' + str(num_layer))
            inputs = self._BatchNormal(inputs, training, name='batch_normal_' + str(num_layer))
            inputs = self._Leaky_Relu(inputs, name='leaky_relu_' + str(num_layer))
            num_layer += 1

            # 1
            inputs = self._Conv2d(inputs, filters=64, kernal_size=[3, 3], stride=(2, 2), name='_Conv2d_' + str(num_layer))
            inputs = self._BatchNormal(inputs, training, name='batch_normal_' + str(num_layer))
            inputs = self._Leaky_Relu(inputs, name='leaky_relu_' + str(num_layer))
            num_layer += 1
            shortcut = inputs

            # 2 - 4
            for _ in range(1):
                inputs = self._Conv2d(inputs, filters=32, kernal_size=[1, 1], stride=(1, 1), name='_Conv2d_' + str(num_layer))
                inputs = self._BatchNormal(inputs, training, name='batch_normal_' + str(num_layer))
                inputs = self._Leaky_Relu(inputs, name='leaky_relu_' + str(num_layer))
                num_layer += 1
                inputs = self._Conv2d(inputs, filters=64, kernal_size=[3, 3], stride=(1, 1), name='_Conv2d_' + str(num_layer))
                inputs = self._BatchNormal(inputs, training, name='batch_normal_' + str(num_layer))
                inputs = self._Leaky_Relu(inputs, name='leaky_relu_' + str(num_layer))
                num_layer += 1
                inputs = self._Residual(inputs, shortcut)

            # 5
            inputs = self._Conv2d(inputs, filters=128, kernal_size=[3, 3], stride=(2, 2), name='_Conv2d_' + str(num_layer))
            num_layer += 1
            
            num_d = 1
            num_t = 1

            for layer in self.net_config:
                print(inputs)
                if layer[0] == 'D':
                    inputs = self._Dense_Block(inputs, layer[1], is_dropout=False, training=training, name='Dense_Block' + str(num_d))
                    num_d += 1
                elif layer[0] == 'T':
                    inputs = self._Transition_Layer(inputs, is_avgpool2d=True, training=training, name='TransitionLayer' + str(num_t))
                    num_t += 1
                elif layer[0] == 'S':
                    if layer[1] == 2:
                        for _ in range(8):
                            inputs = self._BatchNormal(inputs, training, name='batch_normal_' + str(num_layer))
                            inputs = self._Leaky_Relu(inputs, name='leaky_relu_' + str(num_layer))
                            inputs = self._Conv2d(inputs, filters=128, kernal_size=[1, 1], stride=(1, 1), name='_Conv2d_' + str(num_layer))
                            num_layer += 1
                            inputs = self._BatchNormal(inputs, training, name='batch_normal_' + str(num_layer))
                            inputs = self._Leaky_Relu(inputs, name='leaky_relu_' + str(num_layer))
                            inputs = self._Conv2d(inputs, filters=256, kernal_size=[3, 3], stride=(1, 1), name='_Conv2d_' + str(num_layer))
                            num_layer += 1

                        self.scale2 = inputs
                        print(self.scale2)
                    elif layer[1] == 1:
                        for _ in range(8):
                            inputs = self._BatchNormal(inputs, training, name='batch_normal_' + str(num_layer))
                            inputs = self._Leaky_Relu(inputs, name='leaky_relu_' + str(num_layer))
                            inputs = self._Conv2d(inputs, filters=128, kernal_size=[1, 1], stride=(1, 1), name='_Conv2d_' + str(num_layer))
                            num_layer += 1
                            inputs = self._BatchNormal(inputs, training, name='batch_normal_' + str(num_layer))
                            inputs = self._Leaky_Relu(inputs, name='leaky_relu_' + str(num_layer))
                            inputs = self._Conv2d(inputs, filters=256, kernal_size=[3, 3], stride=(1, 1), name='_Conv2d_' + str(num_layer))
                            num_layer += 1

                        self.scale1 = inputs
                        print(self.scale1)
                    else:
                        for _ in range(8):
                            inputs = self._BatchNormal(inputs, training, name='batch_normal_' + str(num_layer))
                            inputs = self._Leaky_Relu(inputs, name='leaky_relu_' + str(num_layer))
                            inputs = self._Conv2d(inputs, filters=128, kernal_size=[1, 1], stride=(1, 1), name='_Conv2d_' + str(num_layer))
                            num_layer += 1
                            inputs = self._BatchNormal(inputs, training, name='batch_normal_' + str(num_layer))
                            inputs = self._Leaky_Relu(inputs, name='leaky_relu_' + str(num_layer))
                            inputs = self._Conv2d(inputs, filters=256, kernal_size=[3, 3], stride=(1, 1), name='_Conv2d_' + str(num_layer))
                            num_layer += 1

                        self.scale0 = inputs
                        print(self.scale0)

    def _detection_layer(self, training=True):
        with tf.name_scope('detection_layer'):
            with tf.variable_scope('scale0'):
                #with tf.device('/gpu:0'):
                self.scale0 = self._BatchNormal(self.scale0, training, name='batch_normal_0')
                self.scale0 = self._Leaky_Relu(self.scale0, name='leaky_relu_0')
                self.scale0 = self._Conv2d(self.scale0, 256, kernal_size=[1, 1], name='conv2d_0')

                self.scale0 = self._BatchNormal(self.scale0, training, name='batch_normal_1')
                self.scale0 = self._Leaky_Relu(self.scale0, name='leaky_relu_1')
                self.scale0 = self._Conv2d(self.scale0, 512, kernal_size=[3, 3], name='conv2d_1')

                self.scale0 = self._BatchNormal(self.scale0, training, name='batch_normal_2')
                self.scale0 = self._Leaky_Relu(self.scale0, name='leaky_relu_2')
                self.scale0 = self._Conv2d(self.scale0, 256, kernal_size=[1, 1], name='conv2d_2')

                self.scale0 = self._BatchNormal(self.scale0, training, name='batch_normal_3')
                self.scale0 = self._Leaky_Relu(self.scale0, name='leaky_relu_3')
                self.scale0 = self._Conv2d(self.scale0, 512, kernal_size=[3, 3], name='conv2d_3')

                self.scale0 = self._BatchNormal(self.scale0, training, name='batch_normal_4')
                self.scale0 = self._Leaky_Relu(self.scale0, name='leaky_relu_4')
                self.scale0 = self._Conv2d(self.scale0, 256, kernal_size=[1, 1], name='conv2d_4')

                layer_final = self._UpSampling2d(self.scale0, 256, kernal_size=[1, 1], strides=(2, 2),
                                               name='_UpSampling2d')

                self.scale0 = self._BatchNormal(self.scale0, training, name='batch_normal_5')
                self.scale0 = self._Leaky_Relu(self.scale0, name='leaky_relu_5')
                self.scale0 = self._Conv2d(self.scale0, 512, kernal_size=[3, 3], name='conv2d_5')
                
                self.scale0 = self._BatchNormal(self.scale0, training, name='batch_normal_6')
                self.scale0 = self._Leaky_Relu(self.scale0, name='leaky_relu_6')
                self.scale0 = self._Conv2d(self.scale0,
                                          filters=(self.num_class + 5) * self.num_anchors,
                                          kernal_size=[1, 1],
                                          stride=(1, 1),
                                          name='_Conv2d_output')

            with tf.variable_scope('scale1'):
                #with tf.device('/gpu:0'):
                self.scale1 = self._BatchNormal(self.scale1, training, name='batch_normal_0')
                self.scale1 = self._Leaky_Relu(self.scale1, name='leaky_relu_0')
                self.scale1 = self._Conv2d(self.scale1, 256, kernal_size=[1, 1], name='conv2d_0')

                self.scale1 = tf.concat([self.scale1, layer_final], 3, name='concat_scale_0_to_scale_1')

                self.scale1 = self._BatchNormal(self.scale1, training, name='batch_normal_1')
                self.scale1 = self._Leaky_Relu(self.scale1, name='leaky_relu_1')
                self.scale1 = self._Conv2d(self.scale1, 512, kernal_size=[3, 3], name='conv2d_1')

                self.scale1 = self._BatchNormal(self.scale1, training, name='batch_normal_2')
                self.scale1 = self._Leaky_Relu(self.scale1, name='leaky_relu_2')
                self.scale1 = self._Conv2d(self.scale1, 256, kernal_size=[1, 1], name='conv2d_2')

                self.scale1 = self._BatchNormal(self.scale1, training, name='batch_normal_3')
                self.scale1 = self._Leaky_Relu(self.scale1, name='leaky_relu_3')
                self.scale1 = self._Conv2d(self.scale1, 512, kernal_size=[3, 3], name='conv2d_3')

                self.scale1 = self._BatchNormal(self.scale1, training, name='batch_normal_4')
                self.scale1 = self._Leaky_Relu(self.scale1, name='leaky_relu_4')
                self.scale1 = self._Conv2d(self.scale1, 256, kernal_size=[1, 1], name='conv2d_4')

                layer_final = self._UpSampling2d(self.scale1, 256, kernal_size=[1, 1], strides=(2, 2),
                                               name='_UpSampling2d')

                self.scale1 = self._BatchNormal(self.scale1, training, name='batch_normal_5')
                self.scale1 = self._Leaky_Relu(self.scale1, name='leaky_relu_5')
                self.scale1 = self._Conv2d(self.scale1, 512, kernal_size=[3, 3], name='conv2d_5')

                self.scale1 = self._BatchNormal(self.scale1, training, name='batch_normal_6')
                self.scale1 = self._Leaky_Relu(self.scale1, name='leaky_relu_6')
                self.scale1 = self._Conv2d(self.scale1,
                                          filters=(self.num_class + 5) * self.num_anchors,
                                          kernal_size=[1, 1],
                                          stride=(1, 1),
                                          name='_Conv2d_output')

            with tf.variable_scope('scale2'):
                #with tf.device('/gpu:0'):
                self.scale2 = self._BatchNormal(self.scale2, training, name='batch_normal_0')
                self.scale2 = self._Leaky_Relu(self.scale2, name='leaky_relu_0')
                self.scale2 = self._Conv2d(self.scale2, 256, kernal_size=[1, 1], name='conv2d_0')

                self.scale2 = tf.concat([self.scale2, layer_final], 3,
                                         name='concat_scale_1_to_scale_2')

                self.scale2 = self._BatchNormal(self.scale2, training, name='batch_normal_1')
                self.scale2 = self._Leaky_Relu(self.scale2, name='leaky_relu_1')
                self.scale2 = self._Conv2d(self.scale2, 512, kernal_size=[3, 3], name='conv2d_1')

                self.scale2 = self._BatchNormal(self.scale2, training, name='batch_normal_2')
                self.scale2 = self._Leaky_Relu(self.scale2, name='leaky_relu_2')
                self.scale2 = self._Conv2d(self.scale2, 256, kernal_size=[1, 1], name='conv2d_2')

                self.scale2 = self._BatchNormal(self.scale2, training, name='batch_normal_3')
                self.scale2 = self._Leaky_Relu(self.scale2, name='leaky_relu_3')
                self.scale2 = self._Conv2d(self.scale2, 512, kernal_size=[3, 3], name='conv2d_3')

                self.scale2 = self._BatchNormal(self.scale2, training, name='batch_normal_4')
                self.scale2 = self._Leaky_Relu(self.scale2, name='leaky_relu_4')
                self.scale2 = self._Conv2d(self.scale2, 256, kernal_size=[1, 1], name='conv2d_4')

                self.scale2 = self._BatchNormal(self.scale2, training, name='batch_normal_5')
                self.scale2 = self._Leaky_Relu(self.scale2, name='leaky_relu_5')
                self.scale2 = self._Conv2d(self.scale2, 512, kernal_size=[3, 3], name='conv2d_5')

                self.scale2 = self._BatchNormal(self.scale2, training, name='batch_normal_6')
                self.scale2 = self._Leaky_Relu(self.scale2, name='leaky_relu_6')
                self.scale2 = self._Conv2d(self.scale2,
                                         filters=(self.num_class + 5) * self.num_anchors,
                                         kernal_size=[1, 1],
                                         stride=(1, 1),
                                         name='_Conv2d_output')

    def _Dense_Block(self, inputs, repeat, is_dropout=True, training=True, name='Dense_block'):
        print('dense_block')
        with tf.variable_scope(name):
            inputs = self._Botteneck_Layer(inputs, is_dropout, training, name='Botteneck_Layer_0')
            mixed = inputs
            print(mixed)

            for i in range(1, repeat):
                inputs = self._Botteneck_Layer(mixed, is_dropout, training, name='Botteneck_Layer_' + str(i))
                mixed = tf.concat([inputs, mixed], axis=3)
                print(mixed)

        print('end')
        return mixed

    def _Botteneck_Layer(self, inputs, is_dropout=True, training=True, name='Botteneck_Layer'):
        with tf.variable_scope(name):
            inputs = self._BatchNormal(inputs, training, name='batch_normal_0')
            inputs = self._Leaky_Relu(inputs, name='leaky_relu_0')
            inputs = self._Conv2d(inputs, 4 * self.growth_rate, kernal_size=[1, 1], name='conv2d_0')
            if is_dropout:
                inputs =self._Dropout(inputs, training, name='dropout_0')

            inputs = self._BatchNormal(inputs, training, name='batch_normal_1')
            inputs = self._Leaky_Relu(inputs, name='leaky_relu_1')
            inputs = self._Conv2d(inputs, self.growth_rate, kernal_size=[3, 3], name='conv2d_1')
            if is_dropout:
                inputs =self._Dropout(inputs, training, name='dropout_1')

        return inputs

    def _Transition_Layer(self, inputs, is_avgpool2d=False, training=True, name='Transition_Layer'):
        with tf.variable_scope(name):
            inputs = self._BatchNormal(inputs, training)
            inputs = self._Leaky_Relu(inputs)
            inputs = self._Conv2d(inputs, int(self.theta * self.growth_rate), kernal_size=[1, 1])

            if is_avgpool2d:
                inputs = self._AvgPool2d(inputs, pool_size=[2, 2], strides=2, name='avg_pool2d')
            else:
                inputs = self._Conv2d(inputs, int(self.theta * self.growth_rate), kernal_size=[1, 1], stride=(2, 2),name='conv2d_pool')

        return inputs

    def _Conv2d(self, inputs, filters, kernal_size, stride=(1, 1), name='conv2d'):
        kernal_init = tf.truncated_normal_initializer(stddev=0.02, dtype=tf.float32)
        inputs = tf.layers.conv2d(inputs, filters, kernal_size, stride,
                                 padding='SAME', kernel_initializer=kernal_init, name=name)
        return inputs

    def _UpSampling2d(self, inputs, filters, kernal_size=(1, 1), strides=(2, 2), name=None):
        layer = tf.layers.conv2d_transpose(inputs, filters, kernal_size, strides, name=name)
        return layer

    def _Leaky_Relu(self, inputs, name='leaky_relu'):
        inputs = tf.maximum(inputs, tf.multiply(inputs, self.alpha), name=name)
        return inputs

    def _Residual(self, conv, shortcut, name='residual'):
        res = self._Leaky_Relu(conv + shortcut, name=name)

        return res

    def _BatchNormal(self, inputs, training, name='batch_normal'):
        inputs = tf.layers.batch_normalization(inputs, training=training, name=name)
        return inputs

    def _AvgPool2d(self, inputs, pool_size, strides, name='avg_pool2d'):
        inputs = tf.layers.average_pooling2d(inputs, pool_size, strides, 'SAME', name=name)
        return inputs
    
    def _MaxPool2d(self, inputs, pool_size, strides, name='max_pool2d'):
        inputs = tf.layers.max_pooling2d(inputs, pool_size, strides, name=name)
        return inputs

    def _Dropout(self, inputs, training, name='dropout'):
        inputs = tf.layers.dropout(inputs, self.drop_rate, training=training, name=name)
        return inputs

    def get_output(self):
        return [self.scale0, self.scale1, self.scale2]

def densenet(inputs_x, training):
    return DenseNet(inputs_x, training)

if __name__ == '__main__':
    inputs = tf.placeholder(tf.float32, (None, 416, 416, 3))

    DenseNet(inputs, training=True)
    print()