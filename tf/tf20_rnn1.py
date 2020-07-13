# 20-07-13_34
# RNN

''' 학습목표
 hihello
 x : hihell  // y : ihello
 h -> i , i -> h , h -> e , l -> l , l -> o 로 학습하는 모델 구현할 것 
 수치화 하고 와꾸를 맞춘다 '''

import tensorflow as tf
import numpy as np
tf.set_random_seed(777)
# 1. data
idx2char = ['e', 'h', 'i', 'l', 'o']                    # 인덱스 넣어주기 위해 한 글씩 뺐다. (알파벳 순?)
_data = np.array([['h', 'i', 'h', 'e', 'l', 'l', 'o']])
# print(_data.shape)                                      # (1, 7)
# print(_data)                                            # [['h' 'i' 'h' 'e' 'l' 'l' 'o']]
# print(type(_data))                                      # <class 'numpy.ndarray'>

''' 데이터 들어가는 것 자체를 onehot 가능
 e 010000
 h 001000
 i 000100
 l 000010
 o 000001 '''

_data = np.array([['h', 'i', 'h', 'e', 'l', 'l', 'o']], dtype=np.str).reshape(-1, 1)

from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
enc.fit(_data)
_data = enc.transform(_data).toarray()

# print('===========')
# print(type(_data))                                      # <class 'numpy.ndarray'> 
# print(_data.dtype)                                      # float64
# print(_data.shape)                                      # 그냥 하면 (1 ,7) => 리쉐이프 (7, 1)로 나와야한다
# print(f'all_data /\n{_data}')

''' _data 형태
 --------------------
 [[0. 1. 0. 0. 0.]  
 ====================
 [0. 0. 1. 0. 0.]      
 [0. 1. 0. 0. 0.]      
 [1. 0. 0. 0. 0.]      
 [0. 0. 0. 1. 0.]      
 [0. 0. 0. 1. 0.]      
 ------------------- x.train
 [0. 0. 0. 0. 1.]] 
 =================== y.train '''

x_data = _data[:6, ]                                    # hihell
y_data = _data[1:, ]                                    # ihello
# print(f'\nx_data / hihell\n{x_data}', x_data.shape)
# print(f'\ny_data / ihello\n{y_data}', y_data.shape)
'''
x_data / hihell
[[0. 1. 0. 0. 0.]
 [0. 0. 1. 0. 0.]
 [0. 1. 0. 0. 0.]
 [1. 0. 0. 0. 0.]
 [0. 0. 0. 1. 0.]
 [0. 0. 0. 1. 0.]] (6, 5) 현재 lstm (1,6,5) 의 x데이터가 준비 되어 있다.

y_data / ihello / y 값들과 대응하는 인덱스일 뿐
[[0. 0. 1. 0. 0.]
 [0. 1. 0. 0. 0.]
 [1. 0. 0. 0. 0.]
 [0. 0. 0. 1. 0.]
 [0. 0. 0. 1. 0.]
 [0. 0. 0. 0. 1.]] (6, 5) 
 기존 keras에서는 (6,) 형태로 y가 설정되어 있었지만, TF1에서는 (1, 6)으로 준비를 해야한다. '''

y_data = np.argmax(y_data, axis=1)
# print(f'\ny_data: {y_data}', '/ shape: ', y_data.shape)                       # y_data: [2 1 0 3 3 4] / shape:  (6,)

x_data = x_data.reshape(1, 6, 5)
y_data = y_data.reshape(1, 6)
# print(f'\nX\n{x_data}\nX.shape: {x_data.shape}\n\nY\n{y_data}\nY.shape: {y_data.shape}')
'''
X
[[[0. 1. 0. 0. 0.]
  [0. 0. 1. 0. 0.]
  [0. 1. 0. 0. 0.]
  [1. 0. 0. 0. 0.]
  [0. 0. 0. 1. 0.]
  [0. 0. 0. 1. 0.]]]
X.shape: (1, 6, 5)

Y
[[2 1 0 3 3 4]]
Y.shape: (1, 6) '''

sequence_length = 6
input_dim = 5
output = 5
batch_size = 1          # 전체 행
lr = 0.1

x = tf.compat.v1.placeholder(tf.float32, (None, sequence_length, input_dim))
y = tf.compat.v1.placeholder(tf.int32, (None, sequence_length))
print(x, '\n', y)
# Tensor("Placeholder:0", shape=(?, 6, 5), dtype=float32) 
#  Tensor("Placeholder_1:0", shape=(?, 6), dtype=int32)


# 2. model
# model.add(LSTM(output, input_shape=(6,5)))
# cell = tf.nn.rnn_cell.BasicLSTMCell(output)                                                       # lstm 의 구조상 피드백 셀 명시
cell = tf.keras.layers.LSTMCell(output)                                                           # 이것 사용해도 된다
hypothesis, _states = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)
print(hypothesis)                                                                                 # Tensor("rnn/transpose_1:0", shape=(?, 6, 5), dtype=float32)


# 3. compile
weights = tf.ones([batch_size, sequence_length])                                                  # 1로 바꿔준다? 선형을 디폴트로 잡고 하겠다.
sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=hypothesis, targets=y, weights=weights)   # 시퀀스 to 시퀀스는 hyothesis와 y값을 빼는 것 뿐
cost = tf.reduce_mean(sequence_loss)                                                              # 전체에 대한 평균
train = tf.compat.v1.train.AdagradOptimizer(learning_rate=lr).minimize(cost)

prediction = tf.argmax(hypothesis, axis=2)      # 1, 6, 5 (3차원에 대한 argmax 5부분을 구해야함으로 axis=2)


# 3-2. train
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(401):
        loss, _, results = sess.run([cost, train, hypothesis], feed_dict={x:x_data, y:y_data})
        pred = sess.run(prediction, feed_dict={x:x_data})
        print(i, f'loss : {loss:.5f}\nhypothesis :\n{results}\npred : {pred}\t true y : {y_data}')

        pred_str = [idx2char[c] for c in np.squeeze(pred)]
        # print(f'\npredict str : ')
        print('\nprediction str :', ''.join(pred_str))
