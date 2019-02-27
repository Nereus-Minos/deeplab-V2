import tensorflow as tf
import numpy as np

from PIL import Image

from deeplab_resnet import DeepLabResNetModel, ImageReader, decode_labels

# 定义获取命令行参数的属性
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
    net = DeepLabResNetModel()

    # 3.预测结果，得到的是灰度图
    pred = net.preds(image_batch)

    # 4加载模型参数

    with tf.Session() as sess:
        restore_var = tf.global_variables()
        loader = tf.train.Saver(var_list=restore_var)
        loader.restore(sess, FLAGS.model_weights)
        print("Restored model parameters from {}".format(FLAGS.model_weights))

        # 5.初始化所有变量
        init_variable = tf.global_variables_initializer()
        sess.run(init_variable)

        # 6.
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)

        # 7.mIoU
        pred = tf.expand_dims(pred, 3)
        predictions = tf.reshape(pred, [-1, ])
        labels = tf.reshape(tf.cast(label_batch, tf.int32), [-1, ])

        mask = labels <= FLAGS.num_classes - 1

        predictions = tf.boolean_mask(predictions, mask)
        labels = tf.boolean_mask(labels, mask)

        # Define the evaluation metric.
        mIoU, update_op = tf.contrib.metrics.streaming_mean_iou(predictions, labels, FLAGS.num_classes)
        # 一定要有
        sess.run(tf.local_variables_initializer())

        # 8.Iterate over images.
        for step in range(FLAGS.num_steps):
            # mIoU_value = sess.run([mIoU])
            # _ = update_op.eval(session=sess)
            preds, _ = sess.run([pred, update_op])

            if FLAGS.save_image_dir is not None:
                img = decode_labels(preds)
                im = Image.fromarray(img[0])
                im.save(FLAGS.save_image_dir + str(step) + '.png')
            if step % 100 == 0:
                print('step {:d} \t'.format(step))
        print('Mean IoU: {:.3f}'.format(mIoU.eval(session=sess)))
        coord.request_stop()
        coord.join(threads)

    return None


if __name__ == '__main__':
    tf.app.run()
