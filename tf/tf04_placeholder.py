# 20-07-07_30
# placeholder

'''
constant는 상수이므로 sess가 필요
tensorflow는 node 연산을 한다
'''


import tensorflow as tf

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
node3 = tf.add(node1, node2)

sess = tf.Session()

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

adder_node = a + b

print(sess.run(adder_node, feed_dict={a:3, b:4.5}))         # adder_node에 feed_dict을 집어 넣는다 (input과 비슷한 개념)
print(sess.run(adder_node, feed_dict={a:[1,3], b:[2,4]}))   # numpy 연산과 비슷하다

add_and_triple = adder_node * 3
print(sess.run(add_and_triple, feed_dict={a:3, b:4.5}))

'''
가중치 연산
        A       B
        O       O
    O       O       O
           OUT
           
sess.run 할 때 feed_dict에서 값을 넣어준다
모든 것을 보여줄 때, sees.run을 이용하고
placeholder에 값을 넣어줄 때는 feed_dict를 이용한다
'''