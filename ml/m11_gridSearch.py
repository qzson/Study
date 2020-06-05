# 20-06-05 // 오후


# 그리드 서치

# 하이퍼파라미터를 최적화하면 모델 성능을 향상시키는데 큰 도움이 됩니다.
# 그리드 서치는 리스트로 지정된 여러 하이퍼파라미터 값을 받아 모든 조합에 대해 모델 성능을 평가하여 최적의 하이퍼파라미터 조합을 찾습니다.
# 그리드 서치와 랜덤 서치를 같이 들어갈 것
# 그 이유는?

# 구성 자체가 모델을 넣고 파라미터라는 리스트에 그 모델에 대한 파라미터 값들이 들어간다
# svc와 rf의 모델 안에 파라미터스에 들어간다.
# 이런 기능을 쓸 수 있다는 것을 알아간다.



import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC


''' 1. 데이터 '''
iris = pd.read_csv('./data/csv/iris.csv', header=0)

x = iris.iloc[:, 0:4]
y = iris.iloc[:, 4]
# print(x)
# print(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=44)

print(x_train.shape)                  # 120, 4
print(x_test.shape)                   # 30, 4
print(y_train.shape)                  # 120,
print(y_test.shape)                   # 30,


parameters =[
    {"C": [1, 10, 100, 1000], "kernel":["linear"]},
    {"C": [1, 10, 100, 1000], "kernel":["rbf"], "gamma":[0.001, 0.0001]},
    {"C": [1, 10, 100, 1000], "kernel":["sigmoid"], "gamma":[0.001, 0.0001]}
]

kfold = KFold(n_splits=5, shuffle=True)
model = GridSearchCV(SVC(), parameters, cv=kfold) # (진짜 모델, 그 모델의 파라미터, 얘를 몇 개씩 하냐)

model.fit(x_train, y_train)

print('최적의 매개변수 :', model.best_estimator_)
y_pred = model.predict(x_test)
print('최종 정답률 :', accuracy_score(y_test, y_pred))