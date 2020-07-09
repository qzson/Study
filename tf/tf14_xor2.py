# 20-07-09_32
# deep learning 구성

import tensorflow as tf
import numpy as np

tf.set_random_seed(777)

x_data = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float32)
y_data = np.array([[0],[1],[1],[0]], dtype=np.float32)
print(x_data.shape)
print(y_data.shape)

x = tf.placeholder(tf.float32, shape=[None, 2])
y = tf.placeholder(tf.float32, shape=[None, 1])

w1 = tf.Variable(tf.random_normal([2, 100]), name='weight1')    # 초기값을 모르기 때문에 random_normal을 사용
b1 = tf.Variable(tf.random_normal([100]), name='bias1')
layer1 = tf.sigmoid(tf.matmul(x, w1) + b1)

w2 = tf.Variable(tf.random_normal([100, 50]), name='weight2')        
b2 = tf.Variable(tf.random_normal([50]), name='bias2')
layer2 = tf.sigmoid(tf.matmul(layer1, w2) + b2)

w3 = tf.Variable(tf.random_normal([50, 1]), name='weight3')        
b3 = tf.Variable(tf.random_normal([1]), name='bias3')
hypothesis = tf.sigmoid(tf.matmul(layer2, w3) + b3)

# sigmoid cost
cost = -tf.reduce_mean(y * tf.log(hypothesis) + (1 - y) * tf.log(1-hypothesis))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1)
train = optimizer.minimize(cost)

# tf.cast, equal 확인
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))

fdt = {x:x_data, y:y_data}

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(3001):
        cost_val, _ = sess.run([cost, train], feed_dict=fdt)

        if step % 20 == 0:
            print(f'{step} cost : {cost_val} \n')
    h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict=fdt)
    print(f'\n Hypothesis :\n {h[0:5]} \n Correct (y) :\n {c[0:5]} \n Acc : {a}')