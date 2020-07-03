# 20-07-03_28
# 소프트맥스 그래프

import numpy as np
import matplotlib.pyplot as plt

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x))

x = np.arange(1,5)
y = softmax(x)

ratio = y
labels = y

plt.pie(ratio,labels = labels, shadow=True, startangle=90)
plt.show()

# softmax분포 다 합치면 1이다.