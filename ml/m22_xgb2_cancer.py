# 200619_24 금요일
# XGB & cancer

from xgboost import XGBClassifier, XGBRegressor, plot_importance
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

x, y = load_breast_cancer(return_X_y=True)
print(x.shape)      # (569, 30)
print(y.shape)      # (569,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8,
                 shuffle = True, random_state = 66)

n_estimators = 150
learning_rate = 0.075
colsample_bytree = 0.9
colsample_bylevel = 0.6
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


# n_estimators = 150
# learning_rate = 0.075
# colsample_bytree = 0.9
# colsample_bylevel = 0.6
# max_depth = 5
# n_jobs = -1
# 점수 : 0.9649122807017544