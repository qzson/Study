# 200604

# KNeighborsClassifier 최근접 이웃

from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

# 1. 데이터
x_data = [[0, 0], [1, 0], [0, 1], [1, 1]]
y_data = [0, 1, 1, 0]

# 2. 모델
# model = SVC()
model = KNeighborsClassifier(n_neighbors=1)
        # 어떤 놈을 이용시킬 것인지 매개변수에 넣어줘야 한다.
        # 가까운 각 매체를 1개씩만 연결하겠다.
        # 2로 하면 acc가 떨어진다. (너무 많이 붙여서? 데이터가 딸랑 4개인데..)

# 3. 훈련
model.fit(x_data, y_data)

# 4. 평가, 예측
x_test = [[0, 0], [1, 0], [0, 1], [1, 1]]
y_predict = model.predict(x_test)

acc = accuracy_score([0, 1, 1, 0], y_predict)

print(x_test, '의 예측 결과', y_predict)
print('acc = ', acc)
