# 20-07-09_32
# preprocessing
# 회귀
import numpy as np
import tensorflow as tf


def min_max_scaler(dataset):
    numerator = dataset - np.min(dataset, 0) # 데이터에서 최솟값을뺀
    denominator = np.max(dataset, 0) - np.min(dataset, 0)
    return numerator / (denominator + 1e-7) # 0으로 나눈다면 그것은 오류 때문에 1e-7을 더해줌 (고정은 아님)

dataset = np.array( [ [828.659973, 833.450012, 908100, 828.349976, 831.659973],
                      [823.02002, 828.070007, 1828100, 821.655029, 828.070007],
                      [819.929993, 824.400024, 1438100, 818.97998, 824.159973],
                      [816, 820.958984, 1008100, 815.48999, 819.23999],
                      [819.359985, 823, 1188100, 818.469971, 818.97998],
                      [819, 823, 1198100, 816, 820.450012],
                      [811.700012, 815.25, 1098100, 809.780029, 813.669983],
                      [809.51001, 816.659973, 1398100, 804.539978, 809.559998],
    ]
)
dataset = min_max_scaler(dataset)
print(dataset)

x_data = dataset[: , 0:-1]
y_data = dataset[: , [-1]] # 여기서는 또 이런식으로 슬라이싱하는데 뭐야

print(x_data.shape) # 8,4
print(y_data.shape) # 8,1

# 회기 모델링
tf.set_random_seed(777)

x_col_num = x_data.shape[1]
y_col_num = y_data.shape[1]

x = tf.placeholder(tf.float32, shape=[None,x_col_num])
y = tf.placeholder(tf.float32, shape=[None,y_col_num])

w = tf.Variable(tf.zeros([x_col_num, y_col_num]), name = 'weight') 
b = tf.Variable(tf.zeros([y_col_num]), name = 'bias') 

h = tf.matmul(x, w) + b 

cost = tf.reduce_mean(tf.square(h - y))

opt = tf.train.GradientDescentOptimizer(learning_rate=0.00002)
train = opt.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    cost_val, h_val, _ = sess.run([cost, h, train], feed_dict={x : x_data, y : y_data})

    if step % 20 == 0:
        print(step, h_val, '\ncost : ', cost_val)

sess.close()