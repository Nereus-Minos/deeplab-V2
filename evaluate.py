import tensorflow as tf
import numpy as np
from PIL import Image

import os

import matplotlib.pyplot as plt

from skimage.color import gray2rgb,rgb2gray

import pydensecrf.densecrf as dcrf      # 安装pip install pydensecrf
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian

# 具体导入请看deeplab_lfov下的__init__.py
# from .model import DeepLabLFOVModel搭建模型
# from .image_reader import ImageReader
# from .utils import decode_labels将灰度图转换为彩色图
from deeplab_resnet import DeepLabResNetModel, ImageReader, decode_labels


IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

# 定义获取命令行参数的名字
FLAGS = tf.app.flags.FLAGS

# 定义命令行参数
tf.app.flags.DEFINE_string("input_size", "321,321", "图片的尺寸（height,width）")
tf.app.flags.DEFINE_integer("batch_size", 16, "一批次数目")
tf.app.flags.DEFINE_string("data_dir", "./data/VOC2012", "数据集目录")
tf.app.flags.DEFINE_string("data_list", "./data/val.txt", "训练集图片名称（方便图片读取）")
tf.app.flags.DEFINE_string("restore_from", "./restore_model/", "restore(预加载)模型参数目录")
tf.app.flags.DEFINE_string("save_image_dir", "./images/eval_save/", "保存预测结果分析图片目录")
tf.app.flags.DEFINE_integer("num_steps", 1449, "使用多少张测试图片做mIoUs")
tf.app.flags.DEFINE_string("model_weights", "./restore_model/deeplab_resnet.ckpt", "模型参数的位置")
tf.app.flags.DEFINE_integer("num_classes", 21, "类别")


def dense_crf(original_image, annotated_image, use_2d=True):

    # 将由FCN等得到的预测结果图annotated_image转换为彩色图，并且统计彩色图中有多少种颜色类别存储在colorize中，并且

    # Converting annotated image to RGB if it is Gray scale
    if (len(annotated_image.shape) < 3):
        annotated_image = gray2rgb(annotated_image)

    # 转换数据类型
    annotated_image = annotated_image.astype(np.int64)

    # Converting the annotations RGB color to single 32 bit integer 即[00000000],[00000000],[00000000]---->[0000000 00000000 00000000]
    annotated_label = annotated_image[:, :, 0] + (annotated_image[:, :, 1] << 8) + (annotated_image[:, :, 2] << 16)

    # Convert the 32bit integer color to 0,1, 2, ... labels.
    # np.unique该函数是去除数组中的重复数字，并进行排序之后输出到colors，
    # return_inverse=True表示返回annotated_label列表元素在colors列表中的位置，并以列表形式储存在label中
    # len(labels)=height*width
    colors, labels = np.unique(annotated_label, return_inverse=True)

    # 防止全部都是0的情况
    if len(colors) == 1 and 0 in colors:
        use_2d = False

    # Creating a mapping back to 32 bit colors ,最后colorize为[[r_1,g_1,b_1],...,[r_m,g_m,b_m],....,[r_n,g_n,b_n]]
    colorize = np.empty((len(colors), 3), np.uint8)
    colorize[:, 0] = (colors & 0x0000FF)
    colorize[:, 1] = (colors & 0x00FF00) >> 8
    colorize[:, 2] = (colors & 0xFF0000) >> 16

    # print(annotated_image.dtype)
    # print(colors)
    # print(np.unique(annotated_image[:, :, 0]))
    # print(np.unique(annotated_image[:, :, 1]))
    # print(np.unique(annotated_image[:, :, 2]))
    # print(np.unique(annotated_image[:, :, 1] << 8))
    # print(np.unique(annotated_image[:, :, 2] << 16))

    # Gives no of class labels in the annotated image
    n_labels = len(set(labels.flat))

    # Setting up the CRF model
    d = dcrf.DenseCRF2D(original_image.shape[1], original_image.shape[0], n_labels)  # width, height, nlabels

    if use_2d:

        # get unary potentials (neg log probability)
        U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=False)
        d.setUnaryEnergy(U)

        # This adds the color-independent term, features are the locations only.
        d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL,
                              normalization=dcrf.NORMALIZE_SYMMETRIC)

        # This adds the color-dependent term, i.e. features are (x,y,r,g,b).
        d.addPairwiseBilateral(sxy=(80, 80), srgb=(13, 13, 13), rgbim=original_image,
                               compat=10,
                               kernel=dcrf.DIAG_KERNEL,
                               normalization=dcrf.NORMALIZE_SYMMETRIC)

    # Run Inference for 5 steps
    Q = d.inference(5)

    # Find out the most probable class for each pixel.
    MAP = np.argmax(Q, axis=0)

    # Convert the MAP (labels) back to the corresponding colors and save the image.
    # Note that there is no "unknown" here anymore, no matter what we had at first.
    MAP = colorize[MAP, :]
    # imsave(output_image, MAP.reshape(original_image.shape))
    return MAP.reshape(original_image.shape)


def main(argv=None):
    # 1.获取测试集数据
    # map()会根据提供的函数对指定序列做映射。FLAGS.input_size.split(',')得到['321','321']
    image_height, image_width = map(int, FLAGS.input_size.split(','))
    input_size = (image_height, image_width)

    # 创建一个线程管理器
    coord = tf.train.Coordinator()

    with tf.name_scope("create_inputs"):
        reader = ImageReader(
            FLAGS.data_dir,
            FLAGS.data_list,
            input_size=None,
            random_scale=False,
            coord=coord)
        image, label = reader.image, reader.label
        image_batch, label_batch = tf.expand_dims(image, dim=0), tf.expand_dims(label, dim=0)

    # 2.建立模型
    net = DeepLabResNetModel(is_training=False, num_classes=FLAGS.num_classes)

    # 3.预测结果
    raw_output_up = net.preds(image_batch)
    #
    pred = tf.expand_dims(raw_output_up, dim=3)

    # 4.定义一个初始化变量op
    init_variable = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init_variable)

        # 5.
        restore_var = tf.global_variables()
        loader = tf.train.Saver(var_list=restore_var)
        loader.restore(sess, FLAGS.model_weights)
        print("Restored model parameters from {}".format(FLAGS.model_weights))

        # 6.
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)

        # 7.mIoU
        predictions = tf.reshape(pred, [-1, ])
        labels = tf.reshape(tf.cast(label_batch, tf.int32), [-1, ])

        mask = labels <= FLAGS.num_classes - 1

        predictions = tf.boolean_mask(predictions, mask)
        labels = tf.boolean_mask(labels, mask)

        # Define the evaluation metric.
        mIoU, update_op = tf.contrib.metrics.streaming_mean_iou(predictions, labels, FLAGS.num_classes)
        # 一定要有
        sess.run(tf.local_variables_initializer())
        # ###

        # 8.Iterate over images.
        for step in range(FLAGS.num_steps):
            # Perform inference.
            preds, img, _ = sess.run([pred, tf.cast(image_batch, tf.uint8), update_op])

            msk = decode_labels(preds, num_classes=FLAGS.num_classes)

            im0 = Image.fromarray(msk[0])

            im0.save(FLAGS.save_image_dir + 'mask' + str(step) + '.png')

            print('The output file has been saved to {}'.format(FLAGS.save_image_dir + 'mask' + str(step) + '.png'))

            # CRF
            raw_im = dense_crf(img[0], msk[0])
            #
            # plt.subplot(1,2,1)
            # plt.imshow(msk[0])
            # plt.subplot(1, 2, 2)
            # plt.imshow(raw_im)
            # plt.show()

            im = Image.fromarray(raw_im)

            im.save(FLAGS.save_image_dir + 'mask_crf' + str(step) + '.png')

            print('The output file has been saved to {}'.format(FLAGS.save_image_dir + 'mask_crf' + str(step) + '.png'))

        print('Mean IoU: {:.3f}'.format(mIoU.eval(session=sess)))

        coord.request_stop()
        coord.join(threads)

    return None


if __name__ == "__main__":
    tf.app.run()

