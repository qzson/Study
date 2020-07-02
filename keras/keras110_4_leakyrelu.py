import numpy as np
import matplotlib.pyplot as plt

def leakyrelu(x):
    return np.maximum(0.01 * x, x)

x = np.arange(-5, 5, 0.1)
y = leakyrelu(x)

plt.plot(x, y)
plt.ylim(-0.3, 1.1)
plt.grid()
plt.show()

# relu 다음 leakyrelu (-0.01까지 살려주는?)
# elu , selu

# 인기순서 relu >> leakyrelu >> elu >> selu