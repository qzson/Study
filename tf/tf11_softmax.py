# 20-07-09_32
# softmax

import tensorflow as tf
import numpy as np


# 1. data
tf.set_random_seed(777)
x_data = [[1, 2, 1, 1],
          [2, 1, 3, 2],
          [3, 1, 3, 4],
          [4, 1, 5, 5],
          [1, 7, 5, 5],
          [1, 2, 5, 6],
          [1, 6, 6, 6],
          [1, 7, 6, 7]]

y_data = [[0, 0, 1],    # 2
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0],    # 1
          [0, 1, 0],
          [0, 1, 0],
          [1, 0, 0],    # 0
          [1, 0, 0]]

x = tf.placeholder('float', shape=[None, 4])
y = tf.placeholder('float', shape=[None, 3])

w = tf.Variable(tf.random_normal([4, 3]), name='weight')    # 행렬 곱 연산 위해 명확히 4,3
# b = tf.Variable(tf.random_normal([3]), name='bias')         # 벡터 3
b = tf.Variable(tf.random_normal([1, 3]), name='bias')      # 벡터 3

# 2. softmax activation
hypothesis = tf.nn.softmax(tf.matmul(x, w) + b)

# 3. categorical cross entropy cost(loss)
cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1))

# 4. gradient descent optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.3).minimize(cost)

# 5. launch(model.fit)
fd = {x: x_data, y: y_data}
pd = {x: [[1, 11, 7, 9]]}
pd2 = {x: [[1, 3, 4, 3]]}
pd3 = {x: [[1, 1, 0, 1]]}
pd4 = {x: [[11, 33, 4, 13]]}
allpd = {x: [[1, 11, 7, 9], [1, 3, 4, 3], [1, 1, 0, 1], [11, 33, 4, 13]]}

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(5001):
        _, cost_val = sess.run([optimizer, cost], feed_dict=fd)

        if step % 500 == 0:
            print(f'{step} cost_v : {cost_val:.5f}')

# 5000 마지막 h 에 값을 집어 넣으면 궁극의 값이 나온다.
# 최적의 w 와 b 가 구해져 있다.

    p = sess.run(hypothesis, feed_dict=pd)
    print('\n',p, '\n pred1 : ',sess.run(tf.argmax(p, 1)))

    p2 = sess.run(hypothesis, feed_dict=pd2)
    print(p2, '\n pred2 : ',sess.run(tf.argmax(p2, 1)))

    p3 = sess.run(hypothesis, feed_dict=pd3)
    print(p3, '\n pred3 : ',sess.run(tf.argmax(p3, 1)))

    p4 = sess.run(hypothesis, feed_dict=pd4)
    print(p4, '\n pred4 : ',sess.run(tf.argmax(p4, 1)))

    all = sess.run(hypothesis, feed_dict=allpd)
    print('\n',all, '\n p1 ~ 4 분류 : ', sess.run(tf.argmax(all, 1)))
    
    # all2 = sess.run(hypothesis, feed_dict={x: [np.append(p, 0), np.append(p2, 0), np.append(p4, 0)]})
    # print('\n',all2, '\n p1 ~ 4 분류 : ', sess.run(tf.argmax(all2, 1))) # 이건 뭐냐