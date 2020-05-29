import numpy as np
from sklearn.datasets import load_breast_cancer

dataset = load_breast_cancer()
x = dataset.data
y = dataset.target

print(x.shape) # (569, 30)
print(y.shape) # (569,)