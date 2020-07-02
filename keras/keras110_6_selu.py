import numpy as np
import matplotlib.pyplot as plt

def selu(x, a =1.67326, l =1.0507):
    y_list = []
    for x in x:
        if x >= 0:
            y = l * x
        if x < 0:
            y = l * a * (np.exp(x) - 1)
        y_list.append(y)
    return y_list

# def selu(x, a = 1.6732, l =1.0507):
    # return list(map(lambda x : l * x if x >= 0 else l*a*(np.exp(x)-1), x))

x = np.arange(-5, 5, 0.1)
y = selu(x)

plt.plot(x, y)
plt.grid()
plt.show()

# relu 다음 leakyrelu (-0.01까지 살려주는?)
# elu , selu

# 인기순서 relu >> leakyrelu >> elu >> selu