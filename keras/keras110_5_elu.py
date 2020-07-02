import numpy as np
import matplotlib.pyplot as plt
# from keras.activations import elu

def elu(x):
    x = np.copy(x)
    x[x<=0]=0.2*(np.exp(x[x<=0])-1)
    return x

x = np.arange(-5, 5, 0.1)
y = elu(x)

plt.plot(x, y)
# plt.ylim(-0.3, 1.1)
plt.grid()
plt.show()

# relu 다음 leakyrelu (-0.01까지 살려주는?)
# elu , selu

# 인기순서 relu >> leakyrelu >> elu >> selu