# 200622_25
# iris, 다중 분류, 완성체


from sklearn.feature_selection import SelectFromModel
import numpy as np
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston, load_breast_cancer, load_iris
from sklearn.metrics import accuracy_score, r2_score


### 데이터 ###
x, y = load_iris(return_X_y=True)
print(x.shape)      # (150, 4)
print(y.shape)      # (150, )

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8,
                 shuffle = True, random_state = 66)


### 모델 ###
model = XGBClassifier(objective='multi:softmax', n_estimators=300, learning_rate=0.1)


### 훈련 ###
model.fit(x_train, y_train, verbose=True, eval_metric=['mlogloss','merror'],
                eval_set=[(x_train, y_train), (x_test, y_test)],
                early_stopping_rounds=20)

# Stopping. Best iteration: [0]
# validation_0-mlogloss:0.97013   validation_0-merror:0.03333
# validation_1-mlogloss:0.97191   validation_1-merror:0.00000


### 평가 ###
results = model.evals_result()
print("eval's results :", results)

y_pred = model.predict(x_test)

acc = accuracy_score(y_pred, y_test)
print('acc :, %.2f%%' %(acc * 100.0))
print('acc :', acc)

# acc :, 100.00%
# acc : 1.0


### 시각화 ###
import matplotlib.pyplot as plt

epochs = len(results['validation_0']['mlogloss'])
x_axis = range(0, epochs)

fig, ax = plt.subplots()
ax.plot(x_axis, results['validation_0']['mlogloss'], label='Train')
ax.plot(x_axis, results['validation_1']['mlogloss'], label='Test')
ax.legend()
plt.ylabel('mLog Loss')
plt.title('XGBoost mLog Loss')

fig, ax = plt.subplots()
ax.plot(x_axis, results['validation_0']['merror'], label='Train')
ax.plot(x_axis, results['validation_1']['merror'], label='Test')
ax.legend()
plt.ylabel('merror')
plt.title('XGBoost merror')
plt.show()
