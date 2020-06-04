# 200604
# LinearSVC 모델로는 해결할 수 없는 xor 연산
# acc 0.5 이 나온다.

from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score

# 1. 데이터
x_data = [[0, 0], [1, 0], [0, 1], [1, 1]]
y_data = [0, 1, 1, 0]

# 2. 모델
model = LinearSVC()

# 3. 훈련
model.fit(x_data, y_data)

# 4. 평가, 예측
x_test = [[0, 0], [1, 0], [0, 1], [1, 1]]
y_predict = model.predict(x_test)

acc = accuracy_score([0, 1, 1, 0], y_predict)

print(x_test, '의 예측 결과', y_predict)
print('acc = ', acc)

'''
 and  0  1    or  0  1    xor  0  1 
 0    0  0    0   0  1    0    0  1
 1    0  1    1   1  1    1    1  0
 
 y = wx + b
 
 x = (4,2) y = (4, )
 and 와 or 둘다 해결 했다
 xor 때문에 인공지능의 암흑기가 왔다.
 '''