# 20-07-07_30
# tensorflow 1 버전
# 3 + 4 + 5
# 4 - 3
# 3 * 4
# 4 / 2


import tensorflow as tf

n0 = tf.constant(2)
n1 = tf.constant(3)
n2 = tf.constant(4)
n3 = tf.constant(5)
add = tf.add_n([n1, n2, n3])
sub = tf.subtract(n2, n1)
mul = tf.multiply(n1, n2)
div = tf.divide(n2, n0)

print('n1 :', n1)       # Tensor("Const:0", shape=(), dtype=int32)
print('\ndiv :', div)   # Tensor("truediv:0", shape=(), dtype=float64)

sess = tf.Session()
print('\nadd :', sess.run(add))
print('sub :', sess.run(sub))
print('mul :', sess.run(mul))
print('div :', sess.run(div))