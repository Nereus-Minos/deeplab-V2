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
from deeplab_resnet import DeepLabResNetModel, decode_labels


IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

# 定义获取命令行参数的名字
FLAGS = tf.app.flags.FLAGS

# 定义命令行参数
tf.app.flags.DEFINE_integer("num_classes", 21, "数据集样本分类种类")
tf.app.flags.DEFINE_string("save_dir", "./images/inference_save/", "预测分割图保存目录")
tf.app.flags.DEFINE_string("img_path", "./images/images_to_inference/test.jpg", "需要预测分割的图片")
tf.app.flags.DEFINE_string("model_weights", "./restore_model/deeplab_resnet.ckpt", "模型参数的位置")


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

    print("No of labels in the Image are ")
    print(n_labels)

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
    # Prepare image.
    img = tf.image.decode_jpeg(tf.read_file(FLAGS.img_path), channels=3)
    # Convert RGB to BGR.
    img_r, img_g, img_b = tf.split(axis=2, num_or_size_splits=3, value=img)
    img = tf.cast(tf.concat(axis=2, values=[img_b, img_g, img_r]), dtype=tf.float32)
    # Extract mean.
    img -= IMG_MEAN
    # 扩展为[batch,h,w,channel]
    img = tf.expand_dims(img, axis=0)

    # 2.建立模型
    net = DeepLabResNetModel(is_training=False, num_classes=FLAGS.num_classes)

    # 3.预测结果
    raw_output_up = net.preds(img)
    #
    print(raw_output_up)
    pred = tf.expand_dims(raw_output_up, dim=3)

    # 4.定义一个初始化变量op
    init_variable = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init_variable)

        restore_var = tf.global_variables()
        # ckpt = tf.train.get_checkpoint_state(FLAGS.model_weights)
        # if ckpt and ckpt.model_checkpoint_path:
        #     loader = tf.train.Saver(var_list=restore_var)
        #     loader.restore(sess, ckpt.model_checkpoint_path)
        #     print("Model restored...")
        loader = tf.train.Saver(var_list=restore_var)
        loader.restore(sess, FLAGS.model_weights)
        print("Restored model parameters from {}".format(FLAGS.model_weights))

        # Perform inference.
        preds, img = sess.run([pred, tf.cast(img, tf.uint8)])

        msk = decode_labels(preds, num_classes=FLAGS.num_classes)

        im0 = Image.fromarray(msk[0])

        if not os.path.exists(FLAGS.save_dir):
            os.makedirs(FLAGS.save_dir)
        im0.save(FLAGS.save_dir + 'mask2.png')

        print('The output file has been saved to {}'.format(FLAGS.save_dir + 'mask2.png'))

        # CRF
        raw_im = dense_crf(img[0], msk[0])

        plt.subplot(1,2,1)
        plt.imshow(msk[0])
        plt.subplot(1, 2, 2)
        plt.imshow(raw_im)
        plt.show()

        im = Image.fromarray(raw_im)

        if not os.path.exists(FLAGS.save_dir):
            os.makedirs(FLAGS.save_dir)
        im.save(FLAGS.save_dir + 'mask3.png')

        print('The output file has been saved to {}'.format(FLAGS.save_dir + 'mask3.png'))

    return None


if __name__ == "__main__":
    tf.app.run()

