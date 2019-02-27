

各文件夹及文件：
  data：数据集文件夹 deeplab_resnet：包含模型（model）、图片读取（image_reder）、及灰度图片上色（utils）
  images：自行测试分割效果的原图（image_to_inference）、自行测试分割图保存（inference_save）、训练过程中保存训练效果图（train_save）
  init_restore_model：训练过程中所使用的fine_tune文件 
  restore_model：最终训练好的模型参数文件 
  save_model：训练过程中保存模型参数文件夹 
  tensorboard：保存tensorboard日志 
  论文：deeplab_v2原文 
  train.py：训练 
  inference.py：分割 
  evaluate.py：评价

要点：
 1.模型基于resnet-101; 
 2.在每层都添加了BN层tf.contrib.slim.batch_norm; 
 3.第四层换成（ conv(11256,1)+BN ---> atrous_conv(33256，2)+BN ---> conv(111024,1)+BN ---> relu ）;
 4.第五层换成（ conv(11512,1)+BN ---> atrous_conv(33512，4)+BN ---> conv(112048,1)+BN ---> relu ）;
 5.使用ASPP(atrous_conv(33num_classes):6、12、18、24)实现感受野多尺度（区别于图片多尺度）; 
 6.在loss中加入了L2正则化损失; 
 7.在做分割图时，使用双线性插值做上采样tf.image.resize_bilinear; 
 8.train：采用自适应学习率 = 基本学习率×（ 1 - iter/max_iter ）^power; 
 9.inference：采用了densecrf做边缘细化; 
 10.evaluate：采用的是mIoU; 
 11.没有使用dropout
