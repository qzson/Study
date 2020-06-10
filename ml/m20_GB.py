# 20-06-10_22 // 16:52~

# 그냥 저냥 쓸만한 놈
# m19 copy

from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

cancer = load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, train_size=0.8, random_state=42
)

# model = DecisionTreeClassifier(max_depth=4)
# model = RandomForestClassifier()
model = GradientBoostingClassifier()

# max_features : 기본값 써라!
# n_estimators : 클수록 좋다!, 단점 메모리 짱 차지, 기본값 100
# n_jobs = -1  : 병렬 처리( *주의 gpu 같이 돌릴땐 2이상 값을 주면 터진다)

model.fit(x_train, y_train)

acc = model.score(x_test, y_test)

print(model.feature_importances_)
print(acc)

import matplotlib.pyplot as plt
import numpy as np
def plot_feature_importances_cancer(model):
    n_features = cancer.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), cancer.feature_names)
    plt.xlabel("Feature Importances")
    plt.ylabel("Features")
    plt.ylim(-1, n_features)

plot_feature_importances_cancer(model)
plt.show()

# [6.62474769e-06 2.03737861e-02 1.83558895e-04 9.88568784e-04
#  2.93848199e-03 2.31205928e-03 2.26971074e-03 4.49354125e-01
#  1.31657537e-03 3.11519215e-04 4.24169865e-03 3.50815617e-03
#  3.91263887e-04 1.02227785e-02 6.22323108e-04 1.70412342e-03
#  1.53819473e-02 2.05696101e-03 7.92768481e-04 1.78910737e-03
#  7.54681866e-02 4.73776208e-02 5.14294506e-02 3.82974278e-02
#  4.80370792e-03 5.10436104e-04 2.02932106e-02 2.39729772e-01
#  1.32230903e-03 1.73999078e-06]
# 0.956140350877193

# acc 바탕으로 나중에 판단을 다 할줄 알아야한다
# 지표 또 다르다