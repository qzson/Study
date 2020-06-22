# 200622_25
# model save & load : pickle


from sklearn.feature_selection import SelectFromModel
import numpy as np
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston, load_breast_cancer
from sklearn.metrics import accuracy_score, r2_score


### 데이터
x, y = load_breast_cancer(return_X_y=True)
print(x.shape)      # (569, 30)
print(y.shape)      # (569,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8,
                 shuffle = True, random_state = 66)


### 모델
model = XGBClassifier(n_estimators=1000, learning_rate=0.1)


### 훈련
model.fit(x_train, y_train, verbose=True, eval_metric='error',
                eval_set=[(x_train, y_train), (x_test, y_test)])


### 평가
results = model.evals_result()
# print("eval's results :", results)

y_pred = model.predict(x_test)

acc = accuracy_score(y_pred, y_test)
print('acc: %.2f%%' %(acc * 100.0))


### pickle 저장 & 불러오기
import pickle # 파이썬 제공 라이브러리
pickle.dump(model, open("./model/xgb_save/cancer.pickle.dat", "wb"))    # wb 라는 형식으로 저장을 하겠다
print("저장됐다.")

model2 = pickle.load(open("./model/xgb_save/cancer.pickle.dat", 'rb'))  #
print("불러왔다.")

y_pred = model2.predict(x_test)

acc = accuracy_score(y_pred, y_test)
print('acc: %.2f%%' %(acc * 100.0))