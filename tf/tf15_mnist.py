# 20-07-09_32
# deep learning _dnn mnist

import tensorflow as tf
import numpy as np
from keras.datasets import mnist
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# 1. data
tf.set_random_seed(777)
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, x_test.shape)  # (60000, 28, 28) (10000, 28, 28)
print(y_train.shape, y_test.shape)  # (60000,) (10000,)

def min_max_scaler(dataset):
    numerator = dataset - np.min(dataset, 0)
    denominator = np.max(dataset, 0) - np.min(dataset, 0)
    return numerator / (denominator + 1e-7)

# 1-0. minmaxscaler
x_train = min_max_scaler(x_train)
x_test = min_max_scaler(x_test)
# print(x_train)
# print(x_test)

x_train = x_train.reshape(-1, 28 * 28)
x_test = x_test.reshape(-1, 28 * 28)
x_shape = (28 * 28)

# 1-1. one hot encoding & to numpy
sess = tf.Session()
y_train = sess.run(tf.one_hot(y_train, 10))
y_test = sess.run(tf.one_hot(y_test, 10))
print(y_train.shape, y_test.shape)  # (60000, 10) (10000, 10)

x = tf.placeholder('float', shape=[None, x_shape])
y = tf.placeholder('float', shape=[None, 10])

w1 = tf.Variable(tf.zeros([x_shape, 256]), name='weight1')
b1 = tf.Variable(tf.zeros([256]), name='bias1')
layer1 = tf.matmul(x, w1) + b1

w2 = tf.Variable(tf.zeros([256, 128]), name='weight2')        
b2 = tf.Variable(tf.zeros([128]), name='bias2')
layer2 = tf.matmul(layer1, w2) + b2

w3 = tf.Variable(tf.zeros([128, 64]), name='weight3')        
b3 = tf.Variable(tf.zeros([64]), name='bias3')
layer3 = tf.matmul(layer2, w3) + b3

w4 = tf.Variable(tf.zeros([64, 32]), name='weight4')        
b4 = tf.Variable(tf.zeros([32]), name='bias4')
layer4 = tf.matmul(layer3, w4) + b4

w5 = tf.Variable(tf.zeros([32, 16]), name='weight5')        
b5 = tf.Variable(tf.zeros([16]), name='bias5')
layer5 = tf.matmul(layer4, w5) + b5

w6 = tf.Variable(tf.zeros([16, 8]), name='weight6')        
b6 = tf.Variable(tf.zeros([8]), name='bias6')
layer6 = tf.matmul(layer5, w6) + b6

w7 = tf.Variable(tf.zeros([8, 4]), name='weight7')        
b7 = tf.Variable(tf.zeros([4]), name='bias7')
layer7 = tf.matmul(layer6, w7) + b7

w8 = tf.Variable(tf.zeros([4, 2]), name='weight8')        
b8 = tf.Variable(tf.zeros([2]), name='bias8')
layer8 = tf.matmul(layer7, w8) + b8

w9 = tf.Variable(tf.zeros([2, 10]), name='weight9')        
b9 = tf.Variable(tf.zeros([10]), name='bias9')
layer9 = tf.matmul(layer8, w9) + b9

w10 = tf.Variable(tf.random_normal([10, 10]), name='weight10')        
b10 = tf.Variable(tf.random_normal([10]), name='bias10')
hypothesis = tf.nn.softmax(tf.matmul(layer9, w10) + b10)

# 3. categorical cross entropy cost(loss)
cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1))

# 4. gradient descent optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=4e-5).minimize(cost)

prediction = tf.argmax(hypothesis, 1)
correct_prediction = tf.equal(prediction, tf.argmax(y_test, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 5. launch(model.fit)
train = {x: x_train, y: y_train}
test = {x: x_test}

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(2001):
        _, cost_val = sess.run([optimizer, cost], feed_dict=train)

        # if step % 500 == 0:
        print(f'{step} cost_v : {cost_val:.5f}')

    p, a = sess.run([hypothesis, accuracy], feed_dict=test)
    print('\n',p, '\n pred : ',sess.run(tf.argmax(p, 1)))
    print('acc : {:.2%}'.format(a))