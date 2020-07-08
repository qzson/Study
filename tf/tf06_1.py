# 20-07-08_31
# placeholder 넣어서 작업

'''
텐서플로는 텐서방식의 연산을 한다고 생각하자 (행렬 연산)

'''


import tensorflow as tf
tf.set_random_seed(777)

# 1차원 뿐만 아니라, 고차원 배열까지 처리가능
x_train = tf.placeholder(tf.float32, shape=[None])  # 일단 쉐이프는 알아서 하겠다.
y_train = tf.placeholder(tf.float32, shape=[None])

fd={x_train:[1,2,3], y_train:[3,5,7]}

# weight(W), bias(b) 노드 생성
W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = x_train * W + b

cost = tf.reduce_mean(tf.square(hypothesis - y_train))

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

with tf.Session() as sess:                      # session을 close 하는 것을 with로 해결
    sess.run(tf.global_variables_initializer()) # initializer 변수들을 선언하겠다 (초기화는 1번만 되는 것)

    for step in range(2001):
        _, cost_val, W_val, b_val = sess.run([train, cost, W, b], feed_dict=fd)

        if step % 20 == 0:
            print(step, cost_val, W_val , b_val)

# predict
    print('\n예측 :', sess.run(hypothesis, feed_dict={x_train:[4]}))
    print('\n예측 :', sess.run(hypothesis, feed_dict={x_train:[5,6]}))
    print('\n예측 :', sess.run(hypothesis, feed_dict={x_train:[6,7,8]}))
