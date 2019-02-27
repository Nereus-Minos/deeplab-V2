import tensorflow as tf


def make_var(name, shape):
    return tf.get_variable(name, shape, trainable=True)


def residual(input, kernel_num_1, stride_h_1, kernel_num_a, stride_h_a, kernel_num_b, stride_h_b,
             kernel_num_c, stride_h_c, name, is_training, is_first_part, have_atrous_conv=False):
    """
    residual部分
    :param input:
    :param kernel_num_1: 第一部分
    :param stride_h_1: 第一部分
    :param kernel_num_a:
    :param stride_h_a:
    :param kernel_num_b:
    :param stride_h_b:
    :param kernel_num_c:
    :param stride_h_c:
    :param name: 例如：2a代表第二层的第一部分
    :param is_training:
    :param is_first_part: 是否为第一部分
    :return:
    """
    bn_branch1 = None

    if is_first_part:
        res_branch1 = conv(input, 1, kernel_num_1, stride_h_1, name='res' + name + '_branch1')
        bn_branch1 = batch_normalization(res_branch1, is_training=is_training, activation_fn=None,
                                         name='bn' + name + '_branch1')
    # ###
    res_branch2a = conv(input, 1, kernel_num_a, stride_h_a, name='res' + name + '_branch2a')
    bn_branch2a = batch_normalization(res_branch2a, is_training=is_training, activation_fn=tf.nn.relu,
                                      name='bn' + name + '_branch2a')
    if have_atrous_conv:
        res_branch2b = atrous_conv(bn_branch2a, 3, kernel_num_b, stride_h_b, name='res' + name + '_branch2b')
    else:
        res_branch2b = conv(bn_branch2a, 3, kernel_num_b, stride_h_b, name='res' + name + '_branch2b')

    bn_branch2b = batch_normalization(res_branch2b, is_training=is_training, activation_fn=tf.nn.relu,
                                      name='bn' + name + '_branch2b')
    res_branch2c = conv(bn_branch2b, 1, kernel_num_c, stride_h_c, name='res' + name + '_branch2c')
    bn_branch2c = batch_normalization(res_branch2c, is_training=is_training, activation_fn=None,
                                      name='bn' + name + '_branch2c')

    if is_first_part:
        res = tf.add_n([bn_branch1, bn_branch2c], name='res' + name)
    else:
        res = tf.add_n([input, bn_branch2c], name='res' + name)

    res_relu = tf.nn.relu(res, name='res' + name + '_relu')

    return res_relu


def conv(input, kernel_h, output_channel, stride_h, name, padding='SAME'):
    """

    :param input:
    :param kernel:
    :param output_channel:
    :param stride:
    :param name:
    :param padding:
    :return:
    """
    input_channel = input.get_shape()[-1]
    with tf.variable_scope(name):
        kernel = make_var('weights', shape=[kernel_h, kernel_h, input_channel, output_channel])
        output = tf.nn.conv2d(input, kernel, [1, stride_h, stride_h, 1], padding=padding)
    return output


def atrous_conv(input, kernel_h, output_channel, dilation, name, padding='SAME', biased=False):
    """

    :param input:
    :param kernel_h:
    :param output_channel:
    :param dilation:
    :param name:
    :param padding:
    :return:
    """
    input_channel = input.get_shape()[-1]
    with tf.variable_scope(name):
        kernel = make_var('weights', shape=[kernel_h, kernel_h, input_channel, output_channel])
        output = tf.nn.atrous_conv2d(input, kernel, dilation, padding=padding)

        if biased:
            biases = make_var('biases', [output_channel])
            output = tf.nn.bias_add(output, biases)

    return output


def batch_normalization(input, name, is_training, activation_fn=None, scale=True):
    with tf.variable_scope(name) as scope:
        output = tf.contrib.slim.batch_norm(
            input,
            activation_fn=activation_fn,
            is_training=is_training,
            updates_collections=None,
            scale=scale,
            scope=scope)
        return output


def max_pool(input, kernel_h, stride_h, name, padding='SAME'):
    return tf.nn.avg_pool(input,
                          ksize=[1, kernel_h, kernel_h, 1],
                          strides=[1, stride_h, stride_h, 1],
                          padding=padding,
                          name=name)


def prepare_label(input_batch, new_size, num_classes):
    """
    [batch,height,width] ----> [batch,pred_height,pred_width] ---> [batch,pred_height,pred_width,n_classes]
    :param input_batch:  input tensor of shape [batch_size H W 1].
    :param new_size:  a tensor with new height and width.
    :return: a tensor of shape [batch_size h w 21] with last dimension comprised of 0's and 1's only.
    """
    # As labels are integer numbers, need to use NN interp.
    input_batch = tf.image.resize_nearest_neighbor(input_batch, new_size)

    # [batch_size,H,W,1] ----> [batch,H,W]
    input_batch = tf.squeeze(input_batch, squeeze_dims=[3])

    # ont_hot编码
    input_batch = tf.one_hot(input_batch, depth=num_classes)

    return input_batch


class DeepLabResNetModel(object):

    def __init__(self, is_training=False, num_classes=21):
        self.is_training = is_training
        self.num_classes = num_classes
        pass

    def _create_network(self, image_batch):

        # resnet模型
        # 第一层
        conv1 = conv(image_batch, 7, 64, 2, name='conv1')
        bn_conv1 = batch_normalization(conv1, is_training=self.is_training, activation_fn=tf.nn.relu,
                                       name='bn_conv1')
        pool1 = max_pool(bn_conv1, 3, 2, name='pool1')

        # 第二层
        # 第二层第一部分residual
        res2a_relu = residual(pool1, 256, 1, 64, 1, 64, 1, 256, 1, name='2a', is_training=self.is_training,
                              is_first_part=True)
        # 第二层第二部分residual
        res2b_relu = residual(res2a_relu, None, None, 64, 1, 64, 1, 256, 1, name='2b', is_training=self.is_training,
                              is_first_part=False)
        # 第二层第三部分residual
        res2c_relu = residual(res2b_relu, None, None, 64, 1, 64, 1, 256, 1, name='2c', is_training=self.is_training,
                              is_first_part=False)

        # 第三层
        # 第三层第一部分
        res3a_relu = residual(res2c_relu, 512, 2, 128, 2, 128, 1, 512, 1, name='3a', is_training=self.is_training,
                              is_first_part=True)
        # 第三层第二部分
        res3b1_relu = residual(res3a_relu, None, None, 128, 1, 128, 1, 512, 1, name='3b1', is_training=self.is_training,
                               is_first_part=False)
        # 第三层第三部分
        res3b2_relu = residual(res3b1_relu, None, None, 128, 1, 128, 1, 512, 1, name='3b2', is_training=self.is_training,
                               is_first_part=False)
        # 第三层第四部分
        res3b3_relu = residual(res3b2_relu, None, None, 128, 1, 128, 1, 512, 1, name='3b3', is_training=self.is_training,
                               is_first_part=False)

        # 第四层
        # 第四层第一部分
        res4a_relu = residual(res3b3_relu, 1024, 1, 256, 1, 256, 2, 1024, 1, name='4a', is_training=self.is_training,
                              have_atrous_conv=True, is_first_part=True)
        # 第四层第二部分
        res4b1_relu = residual(res4a_relu, None, None, 256, 1, 256, 2, 1024, 1, name='4b1', is_training=self.is_training,
                               have_atrous_conv=True, is_first_part=False)
        # 第四层第三部分
        res4b2_relu = residual(res4b1_relu, None, None, 256, 1, 256, 2, 1024, 1, name='4b2', is_training=self.is_training,
                               have_atrous_conv=True, is_first_part=False)
        # 第四层第四部分
        res4b3_relu = residual(res4b2_relu, None, None, 256, 1, 256, 2, 1024, 1, name='4b3', is_training=self.is_training,
                               have_atrous_conv=True, is_first_part=False)
        # 第四层第五部分
        res4b4_relu = residual(res4b3_relu, None, None, 256, 1, 256, 2, 1024, 1, name='4b4', is_training=self.is_training,
                               have_atrous_conv=True, is_first_part=False)
        # 第四层第六部分
        res4b5_relu = residual(res4b4_relu, None, None, 256, 1, 256, 2, 1024, 1, name='4b5', is_training=self.is_training,
                               have_atrous_conv=True, is_first_part=False)
        # 第四层第七部分
        res4b6_relu = residual(res4b5_relu, None, None, 256, 1, 256, 2, 1024, 1, name='4b6', is_training=self.is_training,
                               have_atrous_conv=True, is_first_part=False)
        # 第四层第八部分
        res4b7_relu = residual(res4b6_relu, None, None, 256, 1, 256, 2, 1024, 1, name='4b7', is_training=self.is_training,
                               have_atrous_conv=True, is_first_part=False)
        # 第四层第九部分
        res4b8_relu = residual(res4b7_relu, None, None, 256, 1, 256, 2, 1024, 1, name='4b8', is_training=self.is_training,
                               have_atrous_conv=True, is_first_part=False)
        # 第四层第十部分
        res4b9_relu = residual(res4b8_relu, None, None, 256, 1, 256, 2, 1024, 1, name='4b9', is_training=self.is_training,
                               have_atrous_conv=True, is_first_part=False)
        # 第四层第十一部分
        res4b10_relu = residual(res4b9_relu, None, None, 256, 1, 256, 2, 1024, 1, name='4b10', is_training=self.is_training,
                                have_atrous_conv=True, is_first_part=False)
        # 第四层第十二部分
        res4b11_relu = residual(res4b10_relu, None, None, 256, 1, 256, 2, 1024, 1, name='4b11', is_training=self.is_training,
                                have_atrous_conv=True, is_first_part=False)
        # 第四层第十三部分
        res4b12_relu = residual(res4b11_relu, None, None, 256, 1, 256, 2, 1024, 1, name='4b12', is_training=self.is_training,
                                have_atrous_conv=True, is_first_part=False)
        # 第四层第十四部分
        res4b13_relu = residual(res4b12_relu, None, None, 256, 1, 256, 2, 1024, 1, name='4b13', is_training=self.is_training,
                                have_atrous_conv=True, is_first_part=False)
        # 第四层第十五部分
        res4b14_relu = residual(res4b13_relu, None, None, 256, 1, 256, 2, 1024, 1, name='4b14', is_training=self.is_training,
                                have_atrous_conv=True, is_first_part=False)
        # 第四层第十六部分
        res4b15_relu = residual(res4b14_relu, None, None, 256, 1, 256, 2, 1024, 1, name='4b15', is_training=self.is_training,
                                have_atrous_conv=True, is_first_part=False)
        # 第四层第十七部分
        res4b16_relu = residual(res4b15_relu, None, None, 256, 1, 256, 2, 1024, 1, name='4b16', is_training=self.is_training,
                                have_atrous_conv=True, is_first_part=False)
        # 第四层第十八部分
        res4b17_relu = residual(res4b16_relu, None, None, 256, 1, 256, 2, 1024, 1, name='4b17', is_training=self.is_training,
                                have_atrous_conv=True, is_first_part=False)
        # 第四层第十九部分
        res4b18_relu = residual(res4b17_relu, None, None, 256, 1, 256, 2, 1024, 1, name='4b18', is_training=self.is_training,
                                have_atrous_conv=True, is_first_part=False)
        # 第四层第二十部分
        res4b19_relu = residual(res4b18_relu, None, None, 256, 1, 256, 2, 1024, 1, name='4b19', is_training=self.is_training,
                                have_atrous_conv=True, is_first_part=False)
        # 第四层第二十一部分
        res4b20_relu = residual(res4b19_relu, None, None, 256, 1, 256, 2, 1024, 1, name='4b20', is_training=self.is_training,
                                have_atrous_conv=True, is_first_part=False)
        # 第四层第二十二部分
        res4b21_relu = residual(res4b20_relu, None, None, 256, 1, 256, 2, 1024, 1, name='4b21', is_training=self.is_training,
                                have_atrous_conv=True, is_first_part=False)
        # 第四层第二十三部分
        res4b22_relu = residual(res4b21_relu, None, None, 256, 1, 256, 2, 1024, 1, name='4b22', is_training=self.is_training,
                                have_atrous_conv=True, is_first_part=False)

        # 第五层
        # 第五层第一部分
        res5a_relu = residual(res4b22_relu, 2048, 1, 512, 1, 512, 4, 2048, 1, name='5a', is_training=self.is_training,
                              have_atrous_conv=True, is_first_part=True)
        # 第五层第二部分
        res5b_relu = residual(res5a_relu, None, None, 512, 1, 512, 4, 2048, 1, name='5b', is_training=self.is_training,
                              have_atrous_conv=True, is_first_part=False)
        # 第五层第二部分
        res5c_relu = residual(res5b_relu, None, None, 512, 1, 512, 4, 2048, 1, name='5c', is_training=self.is_training,
                              have_atrous_conv=True, is_first_part=False)

        # ASPP层（即FC层）
        fc1_voc12_c0 = atrous_conv(res5c_relu, 3, self.num_classes, 6, name='fc1_voc12_c0', biased=True)

        fc1_voc12_c1 = atrous_conv(res5c_relu, 3, self.num_classes, 12, name='fc1_voc12_c1', biased=True)

        fc1_voc12_c2 = atrous_conv(res5c_relu, 3, self.num_classes, 18, name='fc1_voc12_c2', biased=True)

        fc1_voc12_c3 = atrous_conv(res5c_relu, 3, self.num_classes, 24, name='fc1_voc12_c3', biased=True)

        fc1_voc12 = tf.add_n([fc1_voc12_c0, fc1_voc12_c1, fc1_voc12_c2, fc1_voc12_c3], name='fc1_voc12')

        return fc1_voc12

    def loss(self, image_batch, label_batch, l2_losses):
        raw_output = self._create_network(tf.cast(image_batch, tf.float32))
        # [batch,height,width,n_classes] ----> [batch*width*height,n_classes]
        prediction = tf.reshape(raw_output, [-1, self.num_classes])

        # label_batch调整尺寸进行one-hot编码，以便使用softmax_cross_entropy_with_logits
        # [batch,height,width,1] ----> [batch,pred_height,pred_width] ---> [batch,pred_height,pred_width,n_classes]
        # raw_output.get_shape()[1:3]得到一个TensorShape([height,width])
        # tf.stack处理得到一维数组[height,width], 通过 raw_output.get_shape()[i].value也可获取形状数组的值height或width
        label_batch = prepare_label(label_batch, tf.stack(raw_output.get_shape()[1:3]), num_classes=self.num_classes)
        # [batch,height,width,n_classes] ----> [batch*width*height,n_classes]
        gt = tf.reshape(label_batch, [-1, self.num_classes])

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=gt))
        # 新加 L2正则损失
        reduced_loss = loss + tf.add_n(l2_losses)

        return reduced_loss

    def preds(self, input_batch):
        raw_output = self._create_network(input_batch)

        raw_output = tf.image.resize_bilinear(raw_output, tf.shape(input_batch)[1:3, ])

        raw_output = tf.argmax(raw_output, axis=3)

        return raw_output
