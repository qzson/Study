# 200619_24
# boston, gridsearch(xgb)


from xgboost import XGBClassifier, XGBRegressor, plot_importance
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt

x, y = load_boston(return_X_y=True)

print(x.shape)      # (506, 13)
print(y.shape)      # (506,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8,
                 shuffle = True, random_state = 66)

parameters = [
    {'n_estimators':[100, 200, 300],
     'learning_rate':[0.1, 0.3, 0.001, 0.01],
     'max_depth':[4,5,6]},
    {'n_estimators':[90, 100, 110],
     'learning_rate':[0.1, 0.001, 0.01],
     'max_depth':[4,5,6],
     'colsample_bytree':[0.6, 0.9, 1]},
    {'n_estimators':[90, 110],
     'learning_rate':[0.1, 0.001, 0.5],
     'max_depth':[4,5,6],
     'colsample_bytree':[0.6, 0.9, 1],
     'colsample_bylevel':[0.6, 0.7, 0.9]},
    {'n_estimators':[90, 150],
     'learning_rate':[0.1, 0.075, 0.5],
     'max_depth':[4,5,6],
     'colsample_bytree':[0.6, 0.9, 1],
     'colsample_bylevel':[0.6, 0.7, 0.9]}
]

model = GridSearchCV(XGBRegressor(), parameters, cv=5, n_jobs = -1)
# n_jobs는 그리드 서치에서 쓰는 것이 좋다. 그러지 않고 xbg 모델 안에도 쓰면 왔다갔다하느라 느려진다
model.fit(x_train, y_train)

print('====etimator====')
print(model.best_estimator_)       # 디폴트로 안나오는 것도 명시
print('====params====')
print(model.best_params_)          # 내가 넣은 것 중 최적만 나온다
print('==============')

score = model.score(x_test, y_test)
print('점수 :', score)

# print(model.feature_importances_) # 먹히지 않는다. >> 이것들은 XGB에서 먹힌다
# plot_importance(model)
# plt.show()


# ====etimator====
# XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=0.6,
#              colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
#              importance_type='gain', interaction_constraints='',
#              learning_rate=0.1, max_delta_step=0, max_depth=6,
#              min_child_weight=1, missing=nan, monotone_constraints='()',
#              n_estimators=110, n_jobs=0, num_parallel_tree=1,
#              objective='reg:squarederror', random_state=0, reg_alpha=0,
#              reg_lambda=1, scale_pos_weight=1, subsample=1, tree_method='exact',
#              validate_parameters=1, verbosity=None)
# ====params====
# {'colsample_bylevel': 0.6, 'colsample_bytree': 1, 'learning_rate': 0.1, 'max_depth': 6, 'n_estimators': 110}
# ==============
# 점수 : 0.9311752979545836

# 손코딩 + 그리드서치
# ====etimator====
# XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=0.6,
#              colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
#              importance_type='gain', interaction_constraints='',
#              learning_rate=0.075, max_delta_step=0, max_depth=6,
#              min_child_weight=1, missing=nan, monotone_constraints='()',
#              n_estimators=150, n_jobs=0, num_parallel_tree=1,
#              objective='reg:squarederror', random_state=0, reg_alpha=0,
#              reg_lambda=1, scale_pos_weight=1, subsample=1, tree_method='exact',
#              validate_parameters=1, verbosity=None)
# ====params====
# {'colsample_bylevel': 0.6, 'colsample_bytree': 1, 'learning_rate': 0.075, 'max_depth': 6, 'n_estimators': 150}
# ==============
# 점수 : 0.9385214607826009