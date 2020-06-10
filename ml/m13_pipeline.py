# 20-06-08_20
# 월요일 // 14:00 ~

# Pipeline
# 전처리의 친구


import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


''' 1. 데이터 '''
iris = load_iris()

x = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 43, shuffle = True)


''' 2. 모델 '''
# model = SVC()

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
                                                                  # 전처리와 model이 한번에 돌아감
# pipe = Pipeline([("scaler", MinMaxScaler()), ('svm', SVC())])     # 쓸 scaler와 model을 명시
pipe = make_pipeline(MinMaxScaler(), SVC())
#   >> Pipeline 과 make_pipeline의 차이
#   : 모델 앞에 명시의 차이 (' ',minmaxscaler...)


pipe.fit(x_train, y_train)

print('acc :', pipe.score(x_test, y_test))