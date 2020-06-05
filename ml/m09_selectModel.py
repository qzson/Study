# 20-06-05
# 클래스파이어 모델들 추출

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators
import warnings

warnings.filterwarnings('ignore')

iris = pd.read_csv('./data/csv/iris.csv', header=0)

x = iris.iloc[:, 0:4]
y = iris.iloc[:, 4]

print(x)
print(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=44)

allAlgorithms = all_estimators(type_filter='classifier') # 클래스파이어 모델들을 다 추출한다

for (name, algorithm) in allAlgorithms:
    model = algorithm()

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print(name, '의 정답률 :', accuracy_score(y_test, y_pred))

import sklearn
print(sklearn.__version__) # 0.22.1 버전에 문제 있어서 출력이 안됨 -> 버전 낮춰야함 0.20.1
