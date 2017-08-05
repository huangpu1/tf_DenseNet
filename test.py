import tensorflow as tf
import numpy as np
tf.set_random_seed(2017)
np.random.seed(2017)

import time
from densenet import DenseNet
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)

is_train = True
batch_size = 10
im_shape = [28, 28, 1]
c_num = 10
k = 32
l = [6,6,6,6]

t0 = time.time()

EPOCH_NUM = 5
TMP_TERM = 100

with tf.Session() as sess:
  densenet = DenseNet(sess, c_num, 
    is_train=is_train, im_shape=im_shape, 
    k=k, l=l, keep_prop=0.9)
  sess.run(tf.global_variables_initializer())

  # Training step
  loss_sum = 0
  acc_sum = 0
  iter_num = EPOCH_NUM * mnist.train.num_examples / batch_size
  for iter_ in range(iter_num):
    image, label = mnist.train.next_batch(batch_size)
    image = image.reshape([batch_size]+im_shape)
    loss, acc = densenet.fit(image, label)
    
    loss_sum += loss
    acc_sum += acc

    if (iter_+1) % TMP_TERM == 0:
      print(" {} : loss - {} / acc - {}".format(
        iter_+1, loss_sum / TMP_TERM, acc_sum / TMP_TERM))
      loss_sum = 0
      acc_sum = 0

  # Evaluation step
  correct_sum = 0
  total_sum = 0
  iter_num = mnist.test.num_examples / batch_size
  for _ in range(iter_num):
    image, label = mnist.test.next_batch(batch_size)
    image = image.reshape([batch_size]+im_shape)
    label_ = densenet.predict(image)

    correct_sum += np.sum(label_ == label)
    total_sum += batch_size
  print(" Evaluation result : accurary - {}".format(
    float(correct_sum) / float(total_sum)))
