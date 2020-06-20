# 20-06-10_22 // 17:00 ~

# XGBoost
# m20 copy

from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

cancer = load_breast_cancer()
x, y = load_breast_cancer(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=42
)
print(x_train.shape)    # (455, 30)
print(x_test.shape)     # (114, 30)
print(y_train.shape)    # (455,)
print(y_test.shape)     # (114,)

# model = DecisionTreeClassifier(max_depth=4)
# model = RandomForestClassifier()
# model = GradientBoostingClassifier()
model = XGBClassifier()

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
    n_features = x.shape[1]                                                     # n_features : column 개수
    plt.barh(np.arange(n_features), model.feature_importances_, align='center') # barh : 가로방향 bar chart
    plt.yticks(np.arange(n_features), cancer.feature_names)                     # tick : 축상의 위치표시 지점
    plt.xlabel("Feature Importances")
    plt.ylabel("Features")
    plt.ylim(-1, n_features)                                                    # y축의 최솟값, 최댓값을 지정 / x는 xlim

plot_feature_importances_cancer(model)
plt.show()

# 파라미터에 대해서 공략해야할 부분은 RF 와 XGB 다
# XGB는 꼭 모델에 넣기를 희망한다 (우승권 모델이기 때문에)
# randomserach 같은 것으로 최적값 잡아도 된다
# 이것도 역시 전처리 필요없다 ; 파이프라인에 안집어 넣어도 된다라는 뜻
# 그래서 여러가지 장점 때문에 대회에서 사용되고 이것을 초반에 돌려놓은게 오랜시간 걸쳐 딥러닝 구성한 결과값보다 좋을 수 있다

# 20-06-10 과제
# 1. dc, rf, gb, xgb 데이콘 71개 컬럼에 피처임포턴스 적용
# 2. dc, rf, gb, xgb 로 데이콘 1번째 fit 할 것 // 기존 점수와 비교
#   >> 이것 중에 좋은 것으로 summit 해서 순위 변동 확인
# mail 로 보낼 것