# 20-08-04 Autoencoder's pca

import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import load_diabetes

dataset = load_diabetes()
X = dataset.data
Y = dataset.target

print(X.shape)  # (442, 10)
print(Y.shape)  # (442,)

pca = PCA(n_components=5)
x2 = pca.fit_transform((X))
pca_evr = pca.explained_variance_ratio_     # '압축'된 컬럼들의 중요도
print(pca_evr)                              # [0.40242142 0.14923182 0.12059623 0.09554764 0.06621856]
print(sum(pca_evr))                         # 0.8340156689459766
# 0.17의 손실을 보았다 그렇지만, mnist에서 0이 차지하는 부분이 80% 인 것을 감안하면 이렇게 압축이 되어도 상관이 없다
# eval과 pred 하고 결과를 보고 판단하여 문제있을 시, 다시 수정해서 확인해보는 작업을 반복한다