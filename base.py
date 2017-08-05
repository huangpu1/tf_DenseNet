import tensorflow as tf

class BaseModule(object):
  def __init__(self, name, is_train):
    self.name = name
    self.reuse = None
    self.is_train = is_train

  def save(self, sess, ckpt_path):
    self.saver.save(sess, ckpt_path)

  def restore(self, sess, ckpt_path):
    self.saver.restore(sess, ckpt_path)

  def __call__(self, *args):
    result = self.build_graph(*args)
    if self.reuse is None:
      self.reuse = True
      self.var_list = tf.contrib.framework.get_variables(self.name)
      self.saver = tf.train.Saver(self.var_list)
    return result

  def build_graph(self, *args):
    raise NotImplementedError("Not implemented yet.")

def conv2d(x, c, k, s, p, is_train, batch_norm, activation):
  use_bias = not batch_norm
  x = tf.layers.conv2d(x, c, k, s, p, use_bias=use_bias)
  if batch_norm:
    x = tf.layers.batch_normalization(x, training=is_train)
  if activation is not None:
    x = activation(x)
  return x

def dense(x, c, is_train, batch_norm, activation):
  use_bias = not batch_norm
  x = tf.layers.dense(x, c, use_bias=use_bias)
  if batch_norm:
    x = tf.layers.batch_normalization(x, training=is_train)
  if activation is not None:
    x = activation(x)
  return x

def densenet_add_layer(x, c, is_train, dropout_kp=None):
  y = x

  x = tf.layers.batch_normalization(x, training=is_train)
  x = tf.nn.relu(x)

  # bottleneck
  x = tf.layers.conv2d(x, c, 1, 1, "SAME", use_bias=False)
  if dropout_kp is not None:
    x = tf.nn.dropout(x, dropout_kp)

  x = tf.layers.batch_normalization(x, training=is_train)
  x = tf.nn.relu(x)

  x = tf.layers.conv2d(x, c, 3, 1, "SAME", use_bias=False)
  if dropout_kp is not None:
    x = tf.nn.dropout(x, dropout_kp)

  x = tf.concat([x, y], 3)
  return x

def densenet_add_transition(x, c, is_train, dropout_kp=None, last=False):

  y = x
  x = tf.layers.batch_normalization(x, training=is_train)
  x = tf.nn.relu(x)
  if last :
    x = tf.nn.avg_pool(x, [1,7,7,1], [1,7,7,1], "VALID")
  else :
    x = tf.layers.conv2d(x, c, 1, 1, "SAME")
    if dropout_kp is not None:
      x = tf.nn.dropout(x, dropout_kp)

    x = tf.nn.avg_pool(x, [1,2,2,1], [1,2,2,1], "SAME")
  return x
