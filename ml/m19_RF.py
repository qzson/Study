# 20-06-10_22 수요일 16:40~

# m18 copy
# 

from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

cancer = load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, train_size=0.8, random_state=42
)

# model = DecisionTreeClassifier(max_depth=4)
model = RandomForestClassifier()

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

# [0.02845371 0.01185224 0.050621   0.04142284 0.00779427 0.01753036
#  0.0480761  0.10977518 0.00353388 0.00816755 0.01565728 0.00454696
#  0.0144728  0.03406954 0.00419653 0.00481259 0.00794965 0.00522532
#  0.00357433 0.00676855 0.09729033 0.02146646 0.0949283  0.14372532
#  0.00978071 0.01889241 0.04528584 0.11890088 0.0154814  0.0057477 ]
# 0.9649122807017544

# 그래프가.. DecisionTree와 또 다르다
# 즉, 맹신하지말아라