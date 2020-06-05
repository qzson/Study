# 20-06-05

# K-fold 교차 검증
# 데이터의 수가 적은 경우에는 이 데이터 중의 일부인 검증 데이터(val)의 수도 적기 때문에 검증(val) 성능의 신뢰도가 떨어진다
# 그렇다고 검증 데이터(val)의 수를 증가시키면, 학습용 데이터(train)의 수가 적어지므로 정상적인 학습(fit)이 되지 않는다
# 이러한 딜레마를 해결하기 위한 검증 방법이 'K-폴드 교차검증 방법이다.

'''

K-폴드 교차검증에서는 다음처럼 학습과 검증을 반복한다.

전체 데이터를 K개의 부분 집합( {D1,D2,⋯,DK} )으로 나눈다.
데이터  {D1,D2,⋯,DK−1} 를 학습용 데이터로 사용하여 회귀분석 모형을 만들고 데이터  {DK} 로 교차검증을 한다.
데이터  {D1,D2,⋯,DK−2,DK} 를 학습용 데이터로 사용하여 회귀분석 모형을 만들고 데이터  {DK−1} 로 교차검증을 한다.
⋮ 
데이터  {D2,⋯,DK} 를 학습용 데이터로 사용하여 회귀분석 모형을 만들고 데이터  {D1} 로 교차검증을 한다.
이렇게 하면 총 K개의 모형과 K개의 교차검증 성능이 나온다. 이 K개의 교차검증 성능을 평균하여 최종 교차검증 성능을 계산한다.

'''

import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators
import warnings

warnings.filterwarnings('ignore')


''' 1. 데이터 '''
iris = pd.read_csv('./data/csv/iris.csv', header=0)

x = iris.iloc[:, 0:4]
y = iris.iloc[:, 4]

print(x)
print(y)
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=44)

kfold = KFold(n_splits=5, shuffle=True)
# 전체 데이터 5로 나누어 1회 훈련 = (20%, 20%, 20%, 20%, 20%)를 총 5번 돈다
# 10으로 나누면, (10%...10%) 총 10번 돈다

allAlgorithms = all_estimators(type_filter='classifier') # 클래스파이어 모델들을 다 추출한다

for (name, algorithm) in allAlgorithms:
    model = algorithm()

    scores = cross_val_score(model, x, y, cv=kfold)
    # 5개씩 자른 것을 훈련할 때마다 계속 스코어를 내주겠다
    print(name, '의 정답률 :')
    print(scores, '\n')
    # 결과값은 5개를 평균을 내던지 최대, 최소로 기준점을 잡던지 알아서 해라
    # 딥러닝 역시 cross_val을 적용한다

import sklearn
print(sklearn.__version__) # 0.22.1 버전에 문제 있어서 출력이 안됨 -> 버전 낮춰야함 0.20.1
