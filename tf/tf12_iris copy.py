# 20-07-09_32
# iris_one_hot

import tensorflow as tf
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# 1. data
tf.set_random_seed(777)
x_data, y_data_n = load_iris(return_X_y=True)
print(x_data.shape) # 150, 4
print(y_data_n.shape) # 150,

# 1-1. one hot encoding & to numpy
sess = tf.Session()
y_data = sess.run(tf.one_hot(y_data_n, 3))
print(type(y_data))                         # <class 'numpy.ndarray'>
print(y_data.shape)

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

prediction = tf.argmax(hypothesis, 1)
correct_prediction = tf.equal(prediction, tf.argmax(y_test, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# accuracy = accuracy_score(hypothesis)

# 5. launch(model.fit)
train = {x: x_train, y: y_train}
test = {x: x_test}

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(5001):
        _, cost_val = sess.run([optimizer, cost], feed_dict=train)

        if step % 500 == 0:
            print(f'{step} cost_v : {cost_val:.5f}')

    p, a = sess.run([hypothesis, accuracy], feed_dict=test)
    print('\n',p, '\n pred : ',sess.run(tf.argmax(p, 1)))
    print('acc : {:.2%}'.format(a))