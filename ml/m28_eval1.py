# 200622_25
# boston, 회귀, 완성체


from sklearn.feature_selection import SelectFromModel
import numpy as np
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.metrics import accuracy_score, r2_score


### 데이터 ###
x, y = load_boston(return_X_y=True)
print(x.shape)      # (506, 13)
print(y.shape)      # (506, )

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8,
                 shuffle = True, random_state = 66)


### 모델 ###
model = XGBRegressor(n_estimators=300, learning_rate=0.1)
# XGBRegressor(gpu_id=0, tree_method='gpu_hist') # gpu 가동

### 훈련 ###
model.fit(x_train, y_train, verbose=True, eval_metric=['logloss','rmse'],
                eval_set=[(x_train, y_train), (x_test, y_test)],
                early_stopping_rounds=20)

# Stopping. Best iteration: [78]
# validation_0-logloss:-791.72449 validation_0-rmse:0.57088
# validation_1-logloss:-799.52972 validation_1-rmse:2.36899


### 평가 ###
results = model.evals_result()
print("eval's results :", results)

y_pred = model.predict(x_test)

r2 = r2_score(y_pred, y_test)
print('r2 Score:, %.2f%%' %(r2 * 100.0))
print('r2 :', r2)

# r2 Score:, 93.18%
# r2 : 0.9317913267732658


### 시각화 ###
import matplotlib.pyplot as plt

epochs = len(results['validation_0']['logloss'])
x_axis = range(0, epochs)

fig, ax = plt.subplots()
ax.plot(x_axis, results['validation_0']['logloss'], label='Train')
ax.plot(x_axis, results['validation_1']['logloss'], label='Test')
ax.legend()
plt.ylabel('Log Loss')
plt.title('XGBoost Log Loss')

fig, ax = plt.subplots()
ax.plot(x_axis, results['validation_0']['rmse'], label='Train')
ax.plot(x_axis, results['validation_1']['rmse'], label='Test')
ax.legend()
plt.ylabel('RMSE')
plt.title('XGBoost RMSE')
plt.show()
