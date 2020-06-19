# 200619_24 금요일
# XGB & iris

from xgboost import XGBClassifier, XGBRegressor, plot_importance
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

x, y = load_iris(return_X_y=True)
# dataset = load_iris()
# x = dataset.data
# y = dataset.target
print(x.shape)      # (150, 4)
print(y.shape)      # (150,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8,
                 shuffle = True, random_state = 66)

n_estimators = 100
learning_rate = 0.01
colsample_bytree = 1
colsample_bylevel = 1
max_depth = 5
n_jobs = -1

model = XGBClassifier(max_depth=max_depth, learning_rate=learning_rate,
                     n_estimators=n_estimators, n_jobs=n_jobs,
                     colsample_bylevel = colsample_bylevel,
                     colsample_bytree = colsample_bytree)

model.fit(x_train, y_train)

score = model.score(x_test, y_test)
print('점수 :', score)

print(model.feature_importances_)

plot_importance(model)
# plt.show()


# n_estimators = 100
# learning_rate = 0.01
# colsample_bytree = 1
# colsample_bylevel = 1
# max_depth = 5
# n_jobs = -1
# 점수 : 1.0