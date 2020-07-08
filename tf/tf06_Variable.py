# 20-07-08_31
# Variable sess


import tensorflow as tf
tf.compat.v1.set_random_seed(777)

# weight(W), bias(b) 노드 생성
W = tf.Variable(tf.random.normal([1]), name='weight')
b = tf.Variable(tf.random.normal([1]), name='bias')
print(W)                                              # <tf.Variable 'weight:0' shape=(1,) dtype=float32_ref>

W = tf.Variable([0.3], tf.float32)
print(W)                                              # <tf.Variable 'Variable:0' shape=(1,) dtype=float32_ref>

# sess 1
sess1 = tf.Session()
sess1.run(tf.global_variables_initializer())
aaa = sess1.run(W)
print('aaa :',aaa)                                    # 0.3
sess1.close()                                         # sess는 메모리를 열어 작업을 하고, 다시 메모리를 닫아주는 프로세스가 필요

# sess 2
sess2 = tf.InteractiveSession()                       # InteractiveSession()을 하게 되면,
sess2.run(tf.global_variables_initializer())
bbb = W.eval()                                        # W.eval()을 해주면 된다.
print('bbb :',bbb,)
sess2.close()

# sess 3
sess = tf.Session()
sess.run(tf.global_variables_initializer())
ccc = W.eval(session=sess)
print('\nccc :',ccc)
sess.close()
                                                      # >> sess 1, 2, 3 방식 둘 다 동일 but 1를 제일 많이 사용
