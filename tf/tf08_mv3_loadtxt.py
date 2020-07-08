# 20-07-08_31
# loadtxt

import tensorflow as tf
import numpy as np
# import warnings
# warnings.tensorflow("ignore")

tf.set_random_seed(777)

dataset = np.loadtxt('./data/csv/data-01-test-score.csv', delimiter=',', dtype=np.float32)
print(dataset.shape)            # (25, 4)
print(type(dataset))

x_data = dataset[:, :-1]
y_data = dataset[:, [-1]]
print(x_data.shape)             # (25, 3)
print(y_data.shape)             # (25, 1)

## 슬라이싱 리스트 안씌워주면
# y_data = y_data.reshape(-1, 1)
# print(y_data.shape)             # (25, )

x = tf.placeholder(tf.float32, shape=[None, 3])
y = tf.placeholder(tf.float32, shape=[None, 1])

w = tf.Variable(tf.random_normal([3, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# hypothesis = tf.add(tf.matmul(x, w), b)
hypothesis = tf.matmul(x, w) + b

cost = tf.reduce_mean(tf.square(hypothesis - y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=4e-5)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

fd = {x:x_data, y:y_data}

for step in range(2001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict=fd)
    if step % 500 == 0:
        print(f'{step}, cost : {cost_val}, \n{hy_val[0:5]}')