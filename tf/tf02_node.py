# 20-07-07_30
# tensorflow 1 버전


import tensorflow as tf

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
node3 = tf.add(node1, node2)

print('node1 :', node1, '\nnode2 :', node2, '\nnode3 :', node3)
# 자료형이 나온다
# node1 : Tensor("Const:0", shape=(), dtype=float32) 
# node2 : Tensor("Const_1:0", shape=(), dtype=float32) 
# node3 : Tensor("Add:0", shape=(), dtype=float32)

sess = tf.Session()
print('sess.run(node1, node2) :', sess.run([node1, node2]))
print('sess.run(node3) :', sess.run(node3))
# sess 적용
# sess.run(node1, node2) : [3.0, 4.0]
# sess.run(node3) : 7.0