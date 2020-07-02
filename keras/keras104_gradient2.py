import numpy as np
import matplotlib.pyplot as plt


# 2차함수의 기울기가 0이되는 작업을 할 것 이다. (그것이 최적의 W, 최소의 Loss 값이니까)
# 그라디언트가 0 ?
f = lambda x : x**2 - 4*x + 6
gradient = lambda x: 2*x - 4

x0 = 0.0
MaxIter = 10
learning_rate = 0.25

print('step\tx\tf(x)')
print('{:02d}\t{:6.5f}\t{:6.5f}'.format(0, x0, f(x0)))
# step    x       f(x)
# 00      0.00000 6.00000

# 경사하강법에 대한 가장 기본적인 쌩 코드
for i in range(MaxIter):                        # range(10) = 0 ~ 9
    x1 = x0 - learning_rate * gradient(x0)      # i = 0 일 때, x1 = 0 - 0.25 * (2 * 0 - 4)       // x1 = 1
    x0 = x1                                     # i = 0 일 때, f(x0) = f(1) = 1^2 - (4 * 1) + 6  // f(x0) = 3

    print('{:02d}\t{:6.5f}\t{:6.5f}'.format(i+1, x0, f(x0)))

# step    x       f(x)
# 01      1.00000 3.00000
# 02      1.50000 2.25000
# 03      1.75000 2.06250
# 04      1.87500 2.01562
# 05      1.93750 2.00391
# 06      1.96875 2.00098
# 07      1.98438 2.00024
# 08      1.99219 2.00006
# 09      1.99609 2.00002
# 10      1.99805 2.00000
