# 20-07-07_30
# tensorflow 1 버전


import tensorflow as tf
print(tf.__version__)

hello = tf.constant('Hello World')  # constant 상수 (바뀌지 않는 수)

print(hello)
# Tensor("Const:0", shape=(), dtype=string)

sess = tf.Session()
print(sess.run(hello))
# 원하는 것을 출력하기 위해선 session을 거쳐야한다