# 200622_25
# boston, xgb_early_stopping


from sklearn.feature_selection import SelectFromModel
import numpy as np
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.metrics import accuracy_score, r2_score

x, y = load_boston(return_X_y=True)
print(x.shape)      # (506, 13)
print(y.shape)      # (506, )

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8,
                 shuffle = True, random_state = 66)


model = XGBRegressor(n_estimators=1000, learning_rate=0.1)

model.fit(x_train, y_train, verbose=True, eval_metric='rmse',
                eval_set=[(x_train, y_train), (x_test, y_test)],
                early_stopping_rounds=20)
# early_stopping 결과 78번에서 끊겼다. (0.57088)
# 20으로 두면, 20번 흔들리고 나서, 20번 시작했던 시점으로 기록한다
# 발리데이션에서 꺽여 올라간다 (실제적으로 validation 부분이 더 중요한데, 테스트 부분에서 꺾이고 있다. 2.36899)

results = model.evals_result()
print("eval's results :", results)

y_pred = model.predict(x_test)

r2 = r2_score(y_pred, y_test)
print('r2 Score:, %.2f%%' %(r2 * 100.0))
print('r2 :', r2)