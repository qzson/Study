# 20-06-05 // 오후

# cancer 적용
# RandomForest 적용
# RandomizedSearch 적용


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


''' 1. 데이터 '''
import numpy as np
from sklearn.datasets import load_breast_cancer
x, y = load_breast_cancer(return_X_y=True)
# print(x[0])
print(x.shape) # (569, 30)
print(y.shape) # (569,)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=44)

print(x_train.shape)                  # 120, 4
print(x_test.shape)                   # 30, 4
print(y_train.shape)                  # 120,
print(y_test.shape)                   # 30,


parameters ={
    'n_estimators' : [100,200],
    'max_depth' : [6, 8, 10, 12],
    'min_samples_leaf' : [3, 5, 7, 10],
    'min_samples_split' : [2, 3, 5, 10],
}

kfold = KFold(n_splits=5, shuffle=True)
model = RandomizedSearchCV(RandomForestClassifier(), parameters, cv=kfold, n_jobs=-1)

model.fit(x_train, y_train)

print('최적의 매개변수 :', model.best_estimator_)
y_pred = model.predict(x_test)
print('최종 정답률 :', accuracy_score(y_test, y_pred))