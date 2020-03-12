# Author: Tao Hu <taohu620@gmail.com>
import cv2
import tensorflow as tf


sess = tf.Session()


label = tf.placeholder(tf.int32, shape=(256, 256))
predict = tf.placeholder(tf.int32, shape=(256, 256))
#tf.Print(label,)  tf.metrics.mean_iou
#mIoU, update_op = tf.contrib.metrics.streaming_mean_iou(label, predict, num_classes=2)

mIoU, update_op = tf.metrics.mean_iou(label, predict, num_classes=2)


sess.run(tf.initialize_local_variables())

miou_list = []
for i in range(0, 704):
    x = cv2.imread("result/{}-label.png".format(i),cv2.IMREAD_GRAYSCALE)
    y = cv2.imread("result/{}-predict.png".format(i),cv2.IMREAD_GRAYSCALE)
    #yield label, predict
    result = sess.run(update_op,feed_dict={label:x,predict:y})
    miou_list.append(mIoU.eval(session=sess))

print("miou = {}".format(sum(miou_list)/len(miou_list)))



