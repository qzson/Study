# 200619_24 금요일
# 과적합 방지
# 1. 훈련데이터량을 늘린다.
# 2. 피처수를 줄인다.

# 3. regularization
#   >> 별로 효과가 없었다 (Dropout과 비슷? - 결과가 비슷)
# feature를 xgb로 건드려 보겠다


from xgboost import XGBClassifier, XGBRegressor, plot_importance
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

x, y = load_boston(return_X_y=True)
print(x.shape)      # (506, 13)
print(y.shape)      # (506, )

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8,
                 shuffle = True, random_state = 66)

n_estimators = 150          # 트리가 100개
learning_rate = 0.075       # 첫시작시 0.01 로 값을 두고 천천히 튜닝해나간다 // 가장 쎈놈 (딥러닝의 loss에 대한 떡밥)
                            # 최소의 손실율을 구하기 위한 단위가 learning rate
colsample_bytree = 0.9      # 디폴트 1 // 개별 의사결정나무 모형에 사용될 변수 갯수를 지정 // 보통 0.5 ~ 1 사용 // 얼마정도 컬럼을 샘플로 쓸건지
colsample_bylevel = 0.6     # 디폴트 1 // 이것은 추가로 지정하는 것이 의문이 든다?

max_depth = 5               # 디폴트 6 // 과적합 방지를 위해 사용 // CV를 사용해서 적절한 값이 제시되어야함 // 보통 3 ~ 10 값이 적용된다 (max_leaf_nodes가 설정되면 max_depth는 무시된다)
n_jobs = -1

# 딥러닝이 아닐경우는 n_jobs 항상 -1을 사용해라
# 랜던포레스트가 앙상블 // 디시전트리는 트리모델
# 랜덤포레스트가 트리모델이 합쳐진 것이다 (트리 모델이 앙상블 된 것)
# Randomforest 에서 조금 업그레이드 된것이 부스팅
# 부스팅 계열에선 트리 계열에서 쓰는 것을 그대로 쓴다 (특성 중, 전처리를 안해도 된다는 점 / 결측치 제거 안해도 된다는 점)
# XGB의 가장 장점 : 속도가 빠르다(앙상블이라서 다른 머신러닝 모델보단 느림)
#                : 결측치를 제거해준다 (NaN으로 되어 있는 것들 / 하지만 판단하에 수동으로 제거 해주고 진행할 수도 있다) - 한번 빠르게 돌릴 때 좋다
# 위의 params들 모두 중요하다 핵심 param들
# 케라스로 바꿔었을 때 learning_rate 요놈 하나로 결과값이 크게 달라질 것 (그만큼 핵심 키워드)
# xgb시, 하이퍼파라미터 건들 것은 저 4개

# CV 꼭 써라 써서 결과치 봐야한다
# feature importance 해야한다

model = XGBRegressor(max_depth=max_depth, learning_rate=learning_rate,
                     n_estimators=n_estimators, n_jobs=n_jobs,
                     colsample_bylevel = colsample_bylevel,
                     colsample_bytree = colsample_bytree)
# model = XGBRegressor() # 디폴트로 돌릴 때

model.fit(x_train, y_train)

score = model.score(x_test, y_test)
print('점수 :', score)

print(model.feature_importances_)

plot_importance(model)
# matplot에서 제공하는 것
plt.show()

# 이번시간 
# param에 대해서 완벽한 정리를 해야한다.

# XGB Default
# 점수 : 0.9221188544655419

# learning_rate 0.07
# 점수 : 0.9369225858652748

# n_estimators = 150
# learning_rate = 0.075
# colsample_bytree = 0.9
# colsample_bylevel = 0.6
# 점수 : 0.9446478525465573