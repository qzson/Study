# 20-07-08_31
# 회귀 diabetes

from sklearn.datasets import load_diabetes
import tensorflow as tf
import numpy as np
import pandas as pd

tf.set_random_seed(777)
x_data, y_data = load_diabetes(return_X_y=True)
# print(x_data.shape)
# print(y_data.shape)
y_data = y_data.reshape(-1, 1)
# print(y_data.shape)

x = tf.placeholder(tf.float32, shape=[None, 10])
y = tf.placeholder(tf.float32, shape=[None, 1])

w = tf.Variable(tf.random_normal([10, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# hypothesis = tf.add(tf.matmul(x, w), b)
hypothesis = tf.matmul(x, w) + b

cost = tf.reduce_mean(tf.square(hypothesis - y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.95)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

fd = {x:x_data, y:y_data}

for step in range(4001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict=fd)
    if step % 200 == 0:
        print(f'{step}, cost : {cost_val}, \n{hy_val[0:5]}')