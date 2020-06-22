# 200622_25
# cancer, 이진 분류, 완성체


from sklearn.feature_selection import SelectFromModel
import numpy as np
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston, load_breast_cancer
from sklearn.metrics import accuracy_score, r2_score


### 데이터 ###
x, y = load_breast_cancer(return_X_y=True)
print(x.shape)      # (569, 30)
print(y.shape)      # (569, )

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8,
                 shuffle = True, random_state = 66)


### 모델 ###
model = XGBClassifier(n_estimators=300, learning_rate=0.1)


### 훈련 ###
model.fit(x_train, y_train, verbose=True, eval_metric=['logloss','error'],
                eval_set=[(x_train, y_train), (x_test, y_test)],
                early_stopping_rounds=20)

# Stopping. Best iteration: [0]
# validation_0-logloss:0.60761    validation_0-error:0.01758
# validation_1-logloss:0.61300    validation_1-error:0.03509


### 평가 ###
results = model.evals_result()
print("eval's results :", results)

y_pred = model.predict(x_test)

acc = accuracy_score(y_pred, y_test)
print('acc :, %.2f%%' %(acc * 100.0))
print('acc :', acc)

# acc :, 96.49%
# acc : 0.9649122807017544


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
ax.plot(x_axis, results['validation_0']['error'], label='Train')
ax.plot(x_axis, results['validation_1']['error'], label='Test')
ax.legend()
plt.ylabel('ERROR')
plt.title('XGBoost ERROR')
plt.show()

