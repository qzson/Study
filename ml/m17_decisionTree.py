# 20-06-10_22 수요일 15:30~

# decisionTree.py

from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, train_size=0.8, random_state=42
)

model = DecisionTreeClassifier(max_depth=4)
# Decisiontree 에서 알아야 할 것 2가지 'max_depth'와 'feature_importance'

model.fit(x_train, y_train)

acc = model.score(x_test, y_test)

print(model.feature_importances_)

### 정리 ###

# <model. feature_importances_>
# [0.         0.05959094 0.         0.         0.         0.
#  0.         0.70458252 0.         0.         0.01221069 0.
#  0.         0.         0.         0.00639525 0.0189077  0.0162341
#  0.         0.         0.05329492 0.         0.05247428 0.
#  0.00940897 0.         0.         0.06690062 0.         0.        ]

# : 0.70458252 부분의 컬럼(8번째)이 가장 중요한 데이터가 모여있음을 확인 할 수 있다
# - DecisionTreeClassifier 와 Randomforest 의 feature importance 를 서로 비교할줄 알아야 한다
#   >> RF에서는 importance 값이 또 달라질 것이다
# - 중요한 데이터 컬럼을 베이스로 데이터 분석을 해나간다
# - 기존에 데이터를 PCA를 통해서 acc가 90점대가 나오면, 그 데이터에 대한 촉이나 느낌을 잡고 작업 할 수도 있으며
#   비슷한 이치로 Feature_importance 또한 데이터 분석을 하는데 용이하게 작용한다


# <트리구조> (주입식으로 외우자)
# 장점 : 전처리가 필요없다 (그냥 주입식으로 외워)
#   >> 띠로띠로띠또 하며 나가는 형식이기 때문에 상관없다
# 단점 : 과적합이 일어난다 (그러므로 나온 수치는 맹신하지 말아라)
#   >> max_depth와 연관성 // 5이상이 되면 더 과적합이 일어난다


# < 다만 의문점 >
# - train_test_split에서 random_state를 다르게 잡고 돌리면 중요도가 다른 컬럼에 잡힌다
#   >> 이 부분에 대해서 선생님은 당연한 결과라고 말씀해주셨지만, 솔직히 납득은 잘 가지 않는다 (그럼 어떻게 해야하는 가? 에 대한 의문)
# - 특징 중요도가 컬럼마다 달라지면의 의미가 있는 것인지?