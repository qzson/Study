# 20-07-13_34
# RNN 모델을 만드시오.

import tensorflow as tf
import numpy as np

dataset = np.array([1,2,3,4,5,6,7,8,9,10])
print(dataset.shape)  # (10, )

def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1):
        subset = seq[i:(i+size)]
        aaa.append([j for j in subset])
    # print(type(aaa))
    return np.array(aaa)

data = split_x(dataset, 5)
print(data, data.shape)
x_train = data[:5, :]
x_test = data[5:, :]
y_train = dataset[-5:]
print(x_train, x_train.shape) # (5, 5)
print(x_test, x_test.shape)   # [[ 6  7  8  9 10]] (1, 5)
print(y_train, y_train.shape) # [ 6  7  8  9 10] (5,)
x_train = x_train.reshape(5, 1, 5)
x_test = x_test.reshape(5, 1, 1)
y_train = y_train.reshape(1, 5)
print(x_train, x_train.shape) # (5, 5, 1)
print(x_test, x_test.shape)   # (5, 1)
print(y_train, y_train.shape) # (5, 1)


sequence_length = 1
input_dim = 5
output = 20
batch_size = 1
lr = 0.1

x = tf.compat.v1.placeholder(tf.float32, (None, sequence_length, input_dim))
y = tf.compat.v1.placeholder(tf.int32, (None, 5))
print(x, '\n', y)
# Tensor("Placeholder:0", shape=(?, 5, 1), dtype=float32) 
#  Tensor("Placeholder_1:0", shape=(?, 5), dtype=int32)


# 2. model
# model.add(LSTM(output, input_shape=(6,5)))
# cell = tf.nn.rnn_cell.BasicLSTMCell(output)                                                       # lstm 의 구조상 피드백 셀 명시
cell = tf.keras.layers.LSTMCell(output)                                                           # 이것 사용해도 된다
hypothesis, _states = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)
print(hypothesis)                                                                                 # Tensor("rnn/transpose_1:0", shape=(?, 5, 1), dtype=float32)


# 3. compile
# weights = tf.ones([batch_size, sequence_length])                                                  # 1로 바꿔준다? 선형을 디폴트로 잡고 하겠다.
# sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=hypothesis, targets=y, weights=weights)   # 시퀀스 to 시퀀스는 hyothesis와 y값을 빼는 것 뿐

cost = tf.reduce_mean(sequence_loss)                                                              # 전체에 대한 평균
train = tf.compat.v1.train.AdagradOptimizer(learning_rate=lr).minimize(cost)

# prediction = tf.argmax(hypothesis, axis=2)      # 1, 6, 5 (3차원에 대한 argmax 5부분을 구해야함으로 axis=2)


# 3-2. train
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(401):
        loss, _, results = sess.run([cost, train, hypothesis], feed_dict={x:x_train, y:y_train})
        # pred = sess.run(prediction, feed_dict={x:x_test})
        # print(i, f'loss : {loss:.5f}\nhypothesis :\n{results}\npred : {pred}\t true y : {y_data}')
        print(i, f'loss : {loss:.5f}\nhypothesis :\n{results}\n true y : {y_train}')

#         pred_str = [idx2char[c] for c in np.squeeze(pred)]
#         # print(f'\npredict str : ')
#         print('\nprediction str :', ''.join(pred_str))
