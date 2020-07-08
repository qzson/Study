# 20-07-08_31
# 이진분류 cancer

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import pandas as pd

tf.set_random_seed(777)
x_data, y_data = load_breast_cancer(return_X_y=True)
print(x_data.shape)
print(y_data.shape)
y_data = y_data.reshape(-1, 1)
print(y_data.shape)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2)


x = tf.placeholder(tf.float32, shape=[None, 30])
y = tf.placeholder(tf.float32, shape=[None, 1])

w = tf.Variable(tf.zeros([30, 1]), name='weight')
b = tf.Variable(tf.zeros([1]), name='bias')

hypothesis = tf.sigmoid(tf.matmul(x, w) + b)

# sigmoid cost
cost = -tf.reduce_mean(y * tf.log(hypothesis) + (1 - y) * tf.log(1-hypothesis))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5e-6)
train = optimizer.minimize(cost)

# tf.cast, equal 확인
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))

fdt = {x:x_train, y:y_train}
fdp = {x:x_test, y:y_test}

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(3001):
        cost_val, _ = sess.run([cost, train], feed_dict=fdt)

        if step % 20 == 0:
            print(f'{step} cost : {cost_val} \n')
    h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict=fdp)
    print(f'\n Hypothesis :\n {h[0:5]} \n Correct (y) :\n {c[0:5]} \n Acc : {a}')