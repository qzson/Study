import numpy as np
import matplotlib.pyplot as plt

def relu(x):
    return np.maximum(0, x)

x = np.arange(-5, 5, 0.1)
y = relu(x)

plt.plot(x, y)
plt.grid()
plt.show()

# relu 다음 leakyrelu (-0.01까지 살려주는?)
# elu , selu

# 인기순서 relu >> leakyrelu >> elu >> selu