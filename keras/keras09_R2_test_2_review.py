# 1. 데이터
import numpy as np
x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([1,2,3,4,5,6,7,8,9,10])
x_test = np.array([11,12,13,14,15])
y_test = np.array([11,12,13,14,15])
x_pred = np.array([16, 17, 18])

# 2. 모델구성
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()

model.add(Dense(15, input_dim = 1))
model.add(Dense(5))
model.add(Dense(1000))
model.add(Dense(5))
model.add(Dense(1000))
model.add(Dense(5))
model.add(Dense(1000))
model.add(Dense(5))
model.add(Dense(1000))
model.add(Dense(5))
model.add(Dense(10))
model.add(Dense(1005))
model.add(Dense(5))
model.add(Dense(1000))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(100))
model.add(Dense(5))
model.add(Dense(10))
model.add(Dense(1))

# 3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=252, batch_size=1)

# 4. 평가, 예측
loss, mse = model.evaluate(x_test, y_test)
print("loss :", loss)
print("mse :", mse)


y_predict = model.predict(x_test)
print(y_predict)

# RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))

# R2 구하기
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)

# 결론 : R2는 RMSE 와 같은 보조지표와 같이 쓴다.

# 과제 : R2를 음수가 아닌 0.5 이하로 줄이기
# 레이어는 인풋과 아웃풋을 포함 5개 이상(히든이 3개 이상), 히든레이어 노드는 레이어당 각각 최소 5개 이상
# batch_size = 1
# epochs = 100 이상
# 데이터 조작 하지말아라

'''

이번에는 강제로 나쁜 모델을 만들어야한다
이 조건을 갖춘 상황에서 R2 스코어가 0.5 이하로 나와야 한다.

모르겠다.
fit을 test값으로 하고 evalu를 train 데이터값으로 했을 때, 그리고 RMSE, R2 구하기에서 test를 train으로 변경
했을 때, R2가 가끔 낮은 수로 보일 때가 있었다. 최소값 0.57...

'''

# 200514 0900~ 과제풀이 (r2_test_1)

'''

밥도 적당히 먹으면 좋은데 지나치게 먹으면 토한다.
R2도 마찬가지?
r2는 회계지표에서 acc와 유사하다 - 정확하게 같은 기능은 아니지만 대강 판단하는 것
자, acc 0 일때 r2 0 // (그래프 -x축 epochs 로 봤을 때)
쭈우욱 올라가다가 이후 떨어지는 경우가 생긴다. (과적합 구간 - 토하기 시작하는 시점)
loss는 첫 시점은 높은 값에서 내려오다 마찬가지로 요동친다.
그렇다면 적당한 기준은 어디일까? > 사람이 정한다.
기준을 정했을 때 그 기준이 맞다고 판단하는 법 > 1. 경험치를 쌓아야한다.
2. 쭈우욱 올라가다가 떨어지는 시점에 'if문 쓰든 뚝 떨어지는 시점에서 멈춰라' 라고 코딩을 한다.
[얼리 스타핑] > 코딩하게 되지 않을까? 구현되어 있다.
- 하지만 이것도 문제가 있다. 당한다음에 작동하는 것.? 이건 아직 이해안됨

'''

# 200514 1030~ [Validation]

'''

분포를 잘 잘라서 면적을 같게 해서 분류 하는 것? -전처리의 일부
취미 - 데이터 전처리 / 특기 - 하이퍼 파라미터 튜닝
업무의 90% 데이터 전처리 / 10% 모델링 // $전처리가 상당히 중요한 부분$

    하이닉스, LG,... /    삼전
            x              y
train     1 ~ 10         1 ~ 10
test     11 ~ 15        11 ~ 15
pred     16 ~ 18

두꺼운 책에 손글씨 예제 , 아이리스 예제 > 공통점은 

<미국의 우편번호 자동인식 관련>
1~ 10 까지가 나와야하는데 acc가 97% 정도면 잘나온건가?
3%에 해당하는 것이 잘 못 갔다는 거네. 단 3%의 오차 . 크다.
자 그러면, 어떻게 되는 거야.
회사에서 모델링했어 97%야  근데 대표가 0.1%를 올리라는 거야.
97% 와 97.1% 차이가 크다.
첫 번째, 모델링으로 조작하는 경우도 있지만,

아까 이미지가 6만 장이라고 얘기했었죠
데이터가 많으면 많을 수록 정확해지지만, 머신은 속도가 느려지겠지
이미지를 증폭할거야 예를 들어, 6이라는 숫자가 있으면 비뚤어진 6도 6이다. (증폭과 변환-이미지를 늘려서하는)

70:30으로 훈련과 평가를 하는데 있어서, 머신이 훈련시킬 때 train에서 하는 데이터 수는 10개 (위표에서)
거기서 test의 데이터는 평가만 하기 때문에 머신이 훈련시 반영을 안한다.
그래서 머신 역시도 같이 비교해보면 좋겠지? 그래서 데이터 셋에 트레인 테스트 이외에 따로 한번더 분리를 해준다.

            x              y
train     1 ~ 7          1 ~ 7
val       8 ~ 10                   :여기까지는 fit에서
test     11 ~ 15        11 ~ 15    :여기에서는 evalu...에서
pred     16 ~ 18               

데이터를 나눠서 (val은 머신의 테스트 셋)
= 훈련하고 검증하고 (훈련하고 답 맞춰보고) 그래서 발리데이션을 별도로 한다.

*val (발리데이션-검증) : 

매트릭스에서 발리데이션을 하게되면 발리데이션 로스가 나오겠지
머신 훈련시키고 발리데이션이 이렇게 됬네? (머신이 답을 미리 컨닝하는 개념)
[훈련시키고 컨닝하고 그러기 때문에 그것에 관한 가중치 값이 훨씬 좋아진다.]
그 비율은 8:1:1로 하는 사람이 있구 7:2:1

아무튼 그 이전에 발리데이션 없이 할 때에는 가중치가 정해져있는데,
발리데이션이 포함되면서 가중치가 변할 수 있다 ?
: 발리데이션이 없으면 교과서 위주로 무작정 달리는 것이지만
검증이라는 발리데이션이 있어서 성능이 훨씬 더 좋아진다.

'''

# 현재까지는 트레인을 나눠서 발리데이션으로 나눠서 검증하고 그거에 따라 가중치가 더 좋아진다는 것을 짚고 가자.

# 200514 11:12~

# fit 안에서 트레인과 발리데이션을 한다는 것 ★

# '크로스 발리데이션' 이거는 나중에 하게 될 것