# 20-05-31 / 일요일

# 이건 이진 분류 같음


### 1. 데이터
import numpy as np
from sklearn.datasets import load_breast_cancer
x, y = load_breast_cancer(return_X_y=True)
# print(x[0])
print(x.shape) # (569, 30)
print(y.shape) # (569,)
