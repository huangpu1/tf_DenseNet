import os
import tensorflow as tf

from base import *

class DenseNet(object):
  def __init__(self, sess, c, 
               is_train=True,
               im_shape=None,
               k=32,
               l=None,
               keep_prop=None,
               optim=None):
    """
      Args:
        sess : 
          TensorFlow session
        c :
          An integer.
          The number of class to classify.

      Optional args:
        is_train : 
          A bool.
          Whether training mode or not.
          The default is True.
        im_shape :
          A 3-length list.
          The defualt is [224, 224, 3]
        k : 
          An integer.
          Growth rate of dense blocks.
        l : 
          An integer or a 4-length list.
          How many layers for each block.
        keep_prop :
          Keep proportion of DropOut.
          The default option is 'not using DropOut.'
        optim :
          TensorFlow Optimizer
          The default is AdamOptimizer(learning_rate=1e-4)
    """
    self.sess = sess
    self.cls = ClsModule("cls", is_train)

    default_im_shape = [224, 224, 3]
    if im_shape is None:
      im_shape = default_im_shape

    self.image = tf.placeholder(
      tf.float32, [None]+im_shape)
    self.label = tf.placeholder(tf.int64, [None])

    if im_shape[:2] != default_im_shape[:2]:
      self.image_ = tf.image.resize_bilinear(
        self.image, default_im_shape[:2])
    else :
      self.image_ = self.image

    if type(l) is int: l = [l]*4
    if type(l) is list:
      if len(l) != 4:
        raise ValueError(
          "l should be an int or a 4-length list")
    if l is None:
      l = [6, 12, 24, 16]

    self.logit = self.cls(self.image_, k, l, c, keep_prop)
    self.result = tf.argmax(self.logit, axis=1)
    self.prob = tf.nn.softmax(self.logit)

    if is_train : 
      self.loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits=self.logit, labels=self.label))

      self.eq = tf.equal(self.result, self.label)
      self.acc = tf.reduce_mean(tf.cast(self.eq, tf.float32))

      if optim :
        self.optim = optim
      else :
        self.optim = tf.train.AdamOptimizer(1e-4)
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

      with tf.control_dependencies(update_ops):
        self.train = self.optim.minimize(
          self.loss, var_list=self.var_list)

  def fit(self, image, label):
    """
      Args :
        image :
          An np.float32 numpy-array with shape [N, H, W, C].
          [H, W, C] should be same with the given im_shape.
        label :
          An np.int64 numpy-array with shape [N].
          It represents the class of the image from 0 to c-1.
      Return : 
        A 2-length tuple (loss, acc).
        loss :
          The mean cross-entropy loss of the input batch.
        acc :
          The mean accurary of the input batch.
    """
    _, loss, acc = self.sess.run(
      [self.train, self.loss, self.acc],
      {self.image:image, self.label:label})
    return loss, acc

  def predict(self, image):
    """
      Args :
        image :
          An np.float32 numpy-array with shape [N, H, W, C].
          [H, W, C] should be same with the given im_shape.
      Return :
        An np.uint64 numpy-array with shape [N]
        Predicted integer values from 0 to c-1.
    """
    return self.sess.run(self.result, {self.image:image})

  def get_prob(self, image):
    """
      Args :
        image :
          An np.float32 numpy-array with shape [N, H, W, C].
          [H, W, C] should be same with the given im_shape.
      Return :
        An np.float32 numpy-array with shape [N, c]
        The values after softmax operation.
    """
    return self.sess.run(self.prob, {self.image:image})

  def save(self, ckpt_dir):
    mkdir(ckpt_dir)
    self.cls.save(self.sess, 
      os.path.join(ckpt_dir, "model.ckpt"))

  def restore(self, ckpt_dir):
    self.cls.restore(self.sess, 
      os.path.join(ckpt_dir, "model.ckpt"))

class ClsModule(BaseModule):
  def build_graph(self, *args):
    x = args[0]
    k = args[1]
    l = args[2]
    c = args[3]
    kp = args[4]
    
    # input should be 224
    with tf.variable_scope(self.name, reuse=self.reuse):
      x = conv2d(x, k, 7, 2, "SAME", self.is_train, True, tf.nn.relu) # 112
      x = tf.nn.max_pool(x, [1,3,3,1], [1,2,2,1], "SAME") # 56

      x = tf.layers.conv2d(x, k, 3, 1, "SAME", use_bias=False)
      if kp is not None:
        x = tf.nn.dropout(x, kp)
      for i in range(l[0]):
        x = densenet_add_layer(x, k, self.is_train, kp)
      x = densenet_add_transition(x, k, self.is_train, kp) # 28

      for i in range(l[1]):
        x = densenet_add_layer(x, k, self.is_train, kp)
      x = densenet_add_transition(x, k, self.is_train, kp) # 14

      for i in range(l[2]):
        x = densenet_add_layer(x, k, self.is_train, kp)
      x = densenet_add_transition(x, k, self.is_train, kp) # 7

      for i in range(l[3]):
        x = densenet_add_layer(x, k, self.is_train, kp)
      x = densenet_add_transition(x, k, self.is_train, last=True) # 1

      x = tf.contrib.layers.flatten(x)
      x = dense(x, 1024, self.is_train, True, tf.nn.relu)
      x = dense(x, c, self.is_train, False, None)

    return x

# MISC
def mkdir(dir_path):
  try :
    os.makedirs(dir_path)
  except : pass

