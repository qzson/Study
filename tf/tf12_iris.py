# 20-07-09_32
# iris_one_hot

import tensorflow as tf
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


# 1. data
tf.set_random_seed(777)
x_data, y_data_n = load_iris(return_X_y=True)
print(x_data.shape) # 150, 4
print(y_data_n.shape) # 150,

# 1-1. one hot encoding
y_data = tf.one_hot(y_data_n, depth=3)
print(y_data)                               # Tensor("one_hot:0", shape=(150, 1), dtype=float32)
print(type(x_data))                         # <class 'numpy.ndarray'>

# 1-2. to numpy
sess = tf.Session()
y_data = y_data.eval(session=sess)
print(type(y_data))                         # <class 'numpy.ndarray'>

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.1)
print(x_train.shape, x_test.shape)          # (135, 4) (15, 4)
print(y_train.shape, y_test.shape)          # (135, 3) (15, 3)

x = tf.placeholder('float', shape=[None, 4])
y = tf.placeholder('float', shape=[None, 3])

w = tf.Variable(tf.random_normal([4, 3]), name='weight')
b = tf.Variable(tf.random_normal([3]), name='bias')

# 2. softmax activation
hypothesis = tf.nn.softmax(tf.matmul(x, w) + b)

# 3. categorical cross entropy cost(loss)
cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1))

# 4. gradient descent optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.3).minimize(cost)

# predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
# accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))

# 5. launch(model.fit)
train = {x: x_train, y: y_train}
test = {x: x_test, y: y_test}

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(5001):
        _, cost_val = sess.run([optimizer, cost], feed_dict=train)

        if step % 500 == 0:
            print(f'{step} cost_v : {cost_val:.5f}')
    # h = sess.run(hypothesis, feed_dict=test)
    # print(f'\n Hypothesis :\n {h[0:5]} \n Correct (y) :\n {c[0:5]} \n Acc : {a}')

    p = sess.run(hypothesis, feed_dict=test)
    print('\n',p, '\n pred1 : ',sess.run(tf.argmax(p, 1)))
    
# 5000 마지막 h 에 값을 집어 넣으면 궁극의 값이 나온다.
# 최적의 w 와 b 가 구해져 있다.

    # p = sess.run(hypothesis, feed_dict=pd)
    # print('\n',p, '\n pred1 : ',sess.run(tf.argmax(p, 1)))

    # p2 = sess.run(hypothesis, feed_dict=pd2)
    # print(p2, '\n pred2 : ',sess.run(tf.argmax(p2, 1)))

    # p3 = sess.run(hypothesis, feed_dict=pd3)
    # print(p3, '\n pred3 : ',sess.run(tf.argmax(p3, 1)))

    # p4 = sess.run(hypothesis, feed_dict=pd4)
    # print(p4, '\n pred4 : ',sess.run(tf.argmax(p4, 1)))

    # all = sess.run(hypothesis, feed_dict=allpd)
    # print('\n',all, '\n p1 ~ 4 분류 : ', sess.run(tf.argmax(all, 1)))