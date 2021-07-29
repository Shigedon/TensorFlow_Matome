# TensorFlow

## Book
[Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/)
- [10.Introduction to Artificial Neural Networks with Keras](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/ch10.html)
- [11.Training Deep Neural Networks](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/ch11.html)
- [12.Custom Models and Training with TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/ch12.html)
- [13.Loading and Preprocessing Data with TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/ch13.html)
- [14.Deep Computer Kision Using Convolutional Neural Networks](https://learning.oreilly.com/library/view/hands-on-machine-learning/9781492032632/ch14.html)
- [15.Processing Sequences Using RNNs and CNNs](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/ch15.html)
- [16.Natural Language Processing with RNNs and Attention](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/ch16.html)
[Source Code](https://github.com/ageron/handson-ml2)

## Document
[TensorFlow Core v.2.5.0](https://www.tensorflow.org/api_docs/python/tf)

## Video
[Coding Tensorflow](https://www.youtube.com/playlist?list=PLQY2H8rRoyvwLbzbnKJ59NkZvQAW9wLbx)

[DeepLearningAI](https://www.youtube.com/channel/UCcIXc5mJsHVYTZR1maL5l9w/playlists)

## Course
[MIT 6.S191 Introduction to Deep Learning](http://introtodeeplearning.com/)
[YouTube Playlist](https://www.youtube.com/playlist?list=PLtBw6njQRU-rwp5__7C0oIVt26ZgjG9NI)

### Video1-Introduction to Deep Learning
[Video](https://youtu.be/njKP3FqW3Sk)
[Slides](http://introtodeeplearning.com/slides/6S191_MIT_DeepLearning_L1.pdf)
'''python
class myDenseLayer(tf.keras.layers.Layer):
  def __init__(self, input_dim, output_dim):
    super().__init__()
  
  # Initialize weights & bias
  self.W = self.add_weight([input_dim, output_dim])
  self.b = self.add_weight([1, output_dim])
  
 def call(self, inputs):
  # Forward propagate the inputs
  z = tf.matmul(inputs, self.W) + self.b
  
  # Feed through a non-linear activation
  output = tf.math.sigmoid(z)
  
  return output
'''

#### Binary Cross Entropy Loss
Cross entropy loss can be used with models that output a probability between 0 and 1
- Binary Cross Entropy Loss with TensorFlow:
```python
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, predictied))
```

#### Mean Squared Error Loss
Mean squared error loss can be used with regression models that output continuous real numbers
```python
loss = tf.reduce_mean(tf.square(tf.subtract(y, predicted)))
```

#### Gradient Descent
1. Initialize weights randomly $~N(0,\sigma^2)$
2. Loop until convergence:
3. Compute gradient, $\frac{\partialJ(W)}{\partialW}
4. Updata weights, $W\,\leftarrow\,W-\ita\frac{\partialJ(W)}{\partialW}
5. Return weights

```python
# Gradient descent in TensorFlow
import tensorfrow as tf

weights = tf.Variable([tf.random.normal()])

while True: # loop forever
  with tf.GradientTape() as g:
    loss = compute_loss(weights)
    gradient = g.gradient(loss, weights)
    
  weights = weights - lr * gradient
```





