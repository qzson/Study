# 200622_25
# boston, xgb_eval


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
                eval_set=[(x_train, y_train), (x_test, y_test)])
# rmse, mae, logloss, error(error가 0.2면 accuracy는 0.8), auc(정확도, 정밀도; accuracy의 친구다)

results = model.evals_result()
print("eval's results :", results)
# 100번 훈련 시켰다 (n_estimators = 100, 나무의 개수는 epochs)
# rmse로 변경하면 변경한대로 나온다 (validation_0은 train의 리스트 validation_1은 test의 리스트)
# 과적합으로 끊길 부분 1000번 돌릴 때, 530번 정도 부터 (earlystopping)

y_pred = model.predict(x_test)

r2 = r2_score(y_pred, y_test)
print('r2 Score:, %.2f%%' %(r2 * 100.0))
print('r2 :', r2)