# 20-07-10_33
# TF1 CNN mnist

import tensorflow as tf
from keras.datasets import mnist
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from sklearn.metrics import accuracy_score


# 1. data
tf.set_random_seed(777)
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 1-1. one-hot
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# 1-2. reshape (to cnn) & minmaxscaler
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32')/255
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32')/255
print(x_train.shape, x_test.shape)                              # (60000, 28, 28, 1) (10000, 28, 28, 1)
print(y_train.shape, y_test.shape)                              # (60000, 10) (10000, 10)

# 1-3. pre_model_params
x_shape = 28, 28, 1
lr = 0.001

x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
y = tf.placeholder(tf.float32, shape=[None, 10])
keep_prob = tf.placeholder(tf.float32)                          # dropout


# 2. model
# keras version : Conv2D(32, (3, 3), input_shape = (28, 28, 1))
w1 = tf.get_variable('w1', shape=[3, 3, 1, 32])                 # [kernel_size, kernel_size, channel, output]
print(w1)                                                       # Conv2D에는 bias 계산이 자동으로 되서 b 따로 명시 안해줘도 됨
l1 = tf.nn.conv2d(x, w1, strides=[1,1,1,1], padding='SAME')
print(l1)                                                       # (?, 28, 28, 32)
l1 = tf.nn.selu(l1)                                             # conv2d 연산 결과가 activation을 통과한다
l1 = tf.nn.max_pool(l1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
print(l1)                                                       # (?, 14, 14, 32)

w2 = tf.get_variable('w2', shape=[3, 3, 32, 64])                # [ksize, ksize, input, output]
l2 = tf.nn.conv2d(l1, w2, strides=[1,1,1,1], padding='SAME')
l2 = tf.nn.selu(l2)
l2 = tf.nn.max_pool(l2, [1,2,2,1], [1,2,2,1], padding='SAME')
print(l2)                                                       # (?, 7, 7, 128)

w3 = tf.get_variable('w3', shape=[3, 3, 64, 128])
l3 = tf.nn.conv2d(l2, w3, strides=[1,1,1,1], padding='SAME')
l3 = tf.nn.selu(l3)
l3 = tf.nn.max_pool(l3, [1,2,2,1], [1,2,2,1], padding='SAME')
print(l3)                                                       # (?, 4, 4, 128)

w4 = tf.get_variable('w4', shape=[3, 3, 128, 64])
l4 = tf.nn.conv2d(l3, w4, strides=[1,1,1,1], padding='SAME')
l4 = tf.nn.selu(l4)
l4 = tf.nn.max_pool(l4, [1,2,2,1], [1,2,2,1], padding='SAME')
print(l4)                                                       # (?, 2, 2, 64)
print('===================================================')

# 2-1. flatten & dense layer & softmax output
l_flat = tf.reshape(l4, [-1, 2*2*64])

dw1 = tf.get_variable('w5', shape=[2*2*64, 64], initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([64]), name = 'bias1')
dl1 = tf.nn.selu(tf.matmul(l_flat, dw1) + b1)

dw2 = tf.get_variable('w6', shape=[64, 32], initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([32]), name = 'bias2')
dl2 = tf.nn.selu(tf.matmul(dl1, dw2) + b2)

dw3 = tf.get_variable('w7', shape=[32, 10], initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([10]), name = 'bias3')
hypothesis = tf.nn.softmax(tf.matmul(dl2, dw3) + b3)


# 3. compile
cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis = 1)) # cross_entropy
optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)


# 4. fit
sess = tf.Session()
sess.run(tf.global_variables_initializer())

training_epochs = 10
batch_size = 100
total_batch = int(len(x_train) / batch_size)                    # 60000 / 100

for epoch in range(training_epochs):
    avg_cost = 0

    for i in range(total_batch):
        start = i * batch_size
        end = start + batch_size

        batch_xs, batch_ys = x_train[start : end], y_train[start : end]
        c, _ = sess.run([cost, optimizer], feed_dict={x : batch_xs, y : batch_ys, keep_prob : 0.9}) # (1 - keep_prob) : 만큼 dropout
        avg_cost += c / total_batch

    print('Epoch :', '%02d' %(epoch + 1), 'Cost : {:.5f}'.format(avg_cost))
print('훈련 끝')


# 5. predict
prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
acc = sess.run(accuracy, feed_dict={x:x_test, y:y_test, keep_prob:0.9})
print(f'Acc : {acc:.2%}')
sess.close()

# Acc : 98.87%