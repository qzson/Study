# 20-07-02_27
# 이전에 optimizer, learning_rate 추가 했었다.
# activation, node 도 추가 할 수 있다.

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


x = np.arange(-5, 5, 0.1)
y = sigmoid(x)

print(x.shape, y.shape)

plt.plot(x, y)
plt.grid()
plt.show()

# 0과 1사이로 수렴이 되는 시그모이드 그래프
# 시그모이드 함수의 원래 목적 (0과 1사이로 하는 것)
# activation의 목적 (가중치를 한정시키는 것) - 떡밥이 있다 ?

# activation(각 레이어의 연산) 그리고 그 다음 레이어에 toss