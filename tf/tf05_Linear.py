# 20-07-07_30
# tf1's compile


import tensorflow as tf
tf.set_random_seed(777)

x_train = [1,2,3]
# y_train = [1,2,3]     # y = 1x + 0  예측가능
y_train = [3,5,7]     # y = 2x + 1

W = tf.Variable(tf.random_normal([1]), name='weight')   # Variable 변수 (초기화를 시켜야한다)
b = tf.Variable(tf.random_normal([1]), name='bias')

# sess = tf.Session()
# sess.run(tf.global_variables_initializer())             # 초기화를 시켜야한다
# print(sess.run(W))                                      # [2.2086694] (random_seed : 777로 고정)

hypothesis = x_train * W + b    # 가설 //  y = wx + b

cost = tf.reduce_mean(tf.square(hypothesis - y_train))     # 모델 짜고 compile -> loss(cost) : mse를 손으로 짜준 것

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)    # optimizer
# cost 최소화되는 지점에 learning_rate를 구하라?

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        _, cost_val, W_val, b_val = sess.run([train, cost, W, b])   # train은 반환하지 않겠다

        if step % 20 == 0:
            print(step, cost_val, W_val , b_val)

# session open 을 하면 close를 해야하는데 with을 사용하면 close를 해준다.
# 연산 값에 대해서 찾아 가는 것 sess.run 에서 train 되는 것
# compile의 optimizer 그리고 compile