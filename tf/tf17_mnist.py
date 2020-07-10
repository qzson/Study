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

# 1-1. 전처리
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# 1-2. 정규화
x_train = x_train.reshape(-1, 28 * 28).astype('float32')/255
x_test = x_test.reshape(-1, 28 * 28).astype('float32')/255
print(x_train.shape, x_test.shape)  # (60000, 784) (10000, 784)
print(y_train.shape, y_test.shape)  # (60000, 10) (10000, 10)

# x_data = np.concatenate((x_train, y_train), axis=1)
# print(x_data.shape)     # (60000, 794)
# x_data = tf.convert_to_tensor(x_data, dtype=tf.float32)
# print(type(x_data))
# print(x_data.shape)

x_shape = (28 * 28)
lr = 0.001
training_epochs = 15
batch_size = 100
total_batch = int(len(x_train) / batch_size) # 60000 / 100

x = tf.placeholder(tf.float32, [None, x_shape])
y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)       # dropout

w1 = tf.get_variable('w1', shape=[x_shape, 128], initializer=tf.contrib.layers.xavier_initializer())  # 위와 같은 것이지만 이것이 더 많이 쓰인다.
b1 = tf.Variable(tf.random_normal([128]))
L1 = tf.nn.selu(tf.matmul(x, w1) + b1)
L1 = tf.nn.dropout(L1, keep_prob=keep_prob)

w2 = tf.get_variable('w2', shape=[128, 64], initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([64]))
L2 = tf.nn.selu(tf.matmul(L1, w2) + b2)
L2 = tf.nn.dropout(L2, keep_prob=keep_prob)

w3 = tf.get_variable('w3', shape=[64, 32], initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([32]))
L3 = tf.nn.selu(tf.matmul(L2, w3) + b3)
L3 = tf.nn.dropout(L3, keep_prob=keep_prob)

w4 = tf.get_variable('w4', shape=[32, 16], initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([16]))
L4 = tf.nn.selu(tf.matmul(L3, w4) + b4)
L4 = tf.nn.dropout(L4, keep_prob=keep_prob)

w5 = tf.get_variable('w5', shape=[16, 10], initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([10]))
hypothesis = tf.nn.softmax(tf.matmul(L4, w5) + b5)

cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1))
optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

def next_batch(num, data, labels):
    '''
    Return a total of `num` random samples and labels. 
    '''
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

for epoch in range(training_epochs):        # 15
    avg_cost = 0

    for i in range(total_batch):            # 600
        # batch_xs, batch_ys = x_train[i * batch_size : (i+1)*batch_size], y_train[i * batch_size : (i+1)*batch_size]
        start = i * batch_size
        end = start + batch_size
        batch_xs, batch_ys = x_train[start:end], y_train[start:end]
        # batch_xs, batch_ys = next_batch(batch_size, x_train, y_train)
        feed_dict = {x:batch_xs, y:batch_ys, keep_prob:0.8}
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c / total_batch
    
    print('Epoch :', '%04d' % (epoch + 1), 'Cost : {:.9f}'.format(avg_cost))
print('훈련 끝')

prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
acc = sess.run(accuracy, feed_dict={x:x_test, y:y_test, keep_prob:0.8})
print(f'Acc : {acc:.2%}')