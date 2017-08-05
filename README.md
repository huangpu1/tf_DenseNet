# tf_DenseNet
Tensorflow implementation of DenseNet for general classification problems

## Usage
Initialization
```python
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
      The default option is 'not using DropOut'
    optim :
      TensorFlow Optimizer
      The default is AdamOptimizer(learning_rate=1e-4)
"""

with tf.Session as sess:
  ...

  densenet = DenseNet(sess, c)

  ...

```

Train & Prediction
```python
"""
Args :
  image :
    An np.float32 numpy-array with shape [N, H, W, C].
    [H, W, C] should be same with the given im_shape.
  label :
    An np.int64 numpy-array with shape [N].
    It represents the class of the image from 0 to c-1.
Return :
  loss :
    The mean cross-entropy loss of the input batch.
  acc :
    The mean accurary of the input batch.
  label :
    An np.uint64 numpy-array with shape [N]
    Predicted integer values from 0 to c-1.
"""

loss, acc = densenet.fit(image, label)
label = densenet.predict(image)
```

## Test
MNIST example
```
python test.py
```

## References
- [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993)
- [DenseNet Torch implementation](https://github.com/liuzhuang13/DenseNet)
- [Densenet TensorFlow implementation](https://github.com/YixuanLi/densenet-tensorflow)
