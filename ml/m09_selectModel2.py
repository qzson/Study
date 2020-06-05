# 20-06-05
# 리그레서 모델들 추출

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.utils.testing import all_estimators
import warnings

warnings.filterwarnings('ignore')

boston = pd.read_csv('./data/csv/boston_house_prices.csv', header=1)

print(boston)
x = boston.iloc[:, 0:13]
y = boston.iloc[:, 13]

print(x)
print(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=44)

allAlgorithms = all_estimators(type_filter='regressor') # 리그레서 모델들을 다 추출한다

for (name, algorithm) in allAlgorithms:
    model = algorithm()

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print(name, '의 R2 :', r2_score(y_test, y_pred))

import sklearn
print(sklearn.__version__) # 0.22.1 버전에 문제 있어서 출력이 안됨 -> 버전 낮춰야함 0.20.1
