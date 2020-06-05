# 20-06-05

# 게마와 함께하는 wine예제

import pandas as pd
import matplotlib.pyplot as plt

# 와인 데이터 읽기
wine = pd.read_csv('./data/csv/winequality-white.csv', sep=';', header=0)

# pandas x, y로 나누기
y = wine['quality']
x = wine.drop('quality', axis=1)

print(x.shape) # (4898, 11)
print(y.shape) # (4898, )

# 분류의 폭을 축소를 시켜보자 (그렇다면 분포의 폭이 기존보다 조금 더 나아지겠지?)
# y 레이블 축소
newlist = []
for i in list(y):
    if i <=4:
        newlist +=[0]
    elif i <=7:
        newlist +=[1]
    else :
        newlist +=[2]
y = newlist
# print(y)

# # groupby
# count_data = wine.groupby('quality')['quality'].count() # (해당 컬럼 내) 퀄리티의 각 개체별로 숫자를 세겠다. (5가 몇개, 6이 몇개 ...)
# count_data.plot()
# plt.show()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(x_train, y_train)
acc = model.score(x_test, y_test)

from sklearn.metrics import accuracy_score
y_pred = model.predict(x_test)
print('acc_score :', accuracy_score(y_test, y_pred))
print('acc       :', acc)

''' wine4 결과 '''
# 데이터 분류 방식에 따라서 acc 가 달라지겠다.
# 물론 케글 같은 대회에서는 이렇게 조정하는 경우가 없다. (대회에선 y값과 분포를 제공한다)
# 실질적으로 실무에서 평가해달라고 오더를 받을 때, 한쪽으로 몰려서 그 쪽으로 수렴되어 결과가 나올 수 있다.
# 그러므로, 평소에 y값의 분포에 대해서 신경을 써야한다.
# 오더가 어떻게 내려지느냐에 따라서 회귀가 될 수도 있고 분류가 될 수도 있다.