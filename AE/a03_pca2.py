# 20-08-04 Autoencoder's pca n_components

import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import load_diabetes

dataset = load_diabetes()
X = dataset.data
Y = dataset.target

print(X.shape)  # (442, 10)
print(Y.shape)  # (442,)

# pca = PCA(n_components=5)
# x2 = pca.fit_transform((X))
# pca_evr = pca.explained_variance_ratio_     # '압축'된 컬럼들의 중요도
# print(pca_evr)                              # [0.40242142 0.14923182 0.12059623 0.09554764 0.06621856]
# print(sum(pca_evr))                         # 0.8340156689459766

pca = PCA()
pca.fit(X)
cumsum = np.cumsum(pca.explained_variance_ratio_)   # 누적 합
print(cumsum)   # [0.40242142 0.55165324 0.67224947 0.76779711 0.83401567 0.89428759 0.94794364 0.99131196 0.99914395 1.        ]
# 수치가 완벽하지는 않지만, %를 잡고 해당 위치만큼의 components를 설정한다

aaa = np.argmax(cumsum >= 0.94) + 1 # 인덱스 0 부터 시작하니 +1
print(cumsum>=0.94) # [False False False False False False  True  True  True  True]
print(aaa)          # n_components는 7이 된다 (94% 이상되는 피쳐의 특성을 사용하고 싶을 때)
