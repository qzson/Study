# 20-07-08_31
# multi_variable data 수정

import tensorflow as tf
tf.set_random_seed(777)

x_data = [[73., 51., 65.],
          [92., 98., 11.],
          [89., 31., 33.],
          [99., 33., 100.],
          [17., 66., 79.]]
y_data = [[152.],
          [185.],
          [180.],
          [205.],
          [142.]]

x = tf.placeholder(tf.float32, shape=[None, 3])
y = tf.placeholder(tf.float32, shape=[None, 1])

w = tf.Variable(tf.random_normal([3, 1]), name='weight')           # 처음에 x와 연산 (5,3) 와 하기 위해서는 (3, 1) 3이 와야한다. 뒤는 상관없지만, 1로 하면 하나가 값으로 나온다.
b = tf.Variable(tf.random_normal([1]), name='bias')                # 5,3 x 3,1 = (5, 1)

hypothesis = tf.add(tf.matmul(x, w), b)
hypothesis = tf.matmul(x, w) + b                                   # 도 가능
# hypothesis = w * x + b                                           # 행렬 연산이 들어가서 w와 x엔 함수적용 해야함

cost = tf.reduce_mean(tf.square(hypothesis - y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=4e-5)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

fd = {x:x_data, y:y_data}

for step in range(2001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict=fd)
    if step % 50 == 0:
        print(f'{step}, cost : {cost_val}, \n{hy_val}')