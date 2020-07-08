# 20-07-08_31
# hypothesis를 구하기
# H = Wx + b
# aaa, bbb, ccc 자리에 각 hypothesis를 구하기


import tensorflow as tf
tf.compat.v1.set_random_seed(777)

# x, weight(W), bias(b) 노드 생성
x = [1.,2.,3.]
W = tf.Variable(([0.3]), tf.float32)
b = tf.Variable(([1.]), tf.float32)
# hypothesis = tf.add_n([tf.multiply(W, x), b])
hypothesis = W * x + b
print(W)
# print(hypothesis)


# sess 1
sess1 = tf.Session()
sess1.run(tf.global_variables_initializer())
aaa = sess1.run(hypothesis)
print('aaa :',aaa)
sess1.close()

# sess 2
sess2 = tf.InteractiveSession()
sess2.run(tf.global_variables_initializer())
bbb = hypothesis.eval()
print('bbb :',bbb,)
sess2.close()

# sess 3
sess = tf.Session()
sess.run(tf.global_variables_initializer())
ccc = hypothesis.eval(session=sess)
print('\nccc :',ccc)
sess.close()
