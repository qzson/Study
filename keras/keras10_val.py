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
3%에 해당하는 것이 잘 못 갔다는 거네. 단 3%의 오차, 크다.
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

# keras09_test에서는 일부로 나쁜 모델을 만들어 실험을 했으니, 이번 keras10에서는 좋은 모델을 다시 만들어보자

# 1. 데이터
import numpy as np
x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([1,2,3,4,5,6,7,8,9,10])
x_test = np.array([11,12,13,14,15])
y_test = np.array([11,12,13,14,15])
# x_pred = np.array([16, 17, 18])
x_val = np.array([101, 102, 103, 104, 105])
y_val = np.array([101, 102, 103, 104, 105])
'''
# 총 데이터 수는 20개 / w값은 몇 이야? 1일 것 같지? 아님. 통밥으로 얘기해줄 수 있는 사람 없을 걸 (val 때문에)
# 현 데이터 값은 [1~15,101~105](=20개) 인 것. 이걸 그래프로 표현하면 w = 1로 표현되다가 더 급격하게 올라간다. 붙어있는 데이터
'''

# 2. 모델구성
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()

model.add(Dense(5, input_dim = 1))
model.add(Dense(25))
model.add(Dense(50))
model.add(Dense(250))
model.add(Dense(500))
model.add(Dense(250))
model.add(Dense(50))
model.add(Dense(25))
model.add(Dense(5))
model.add(Dense(1)) # RMSE : 8.844011779304222e-06 // R2 : 0.9999999999608917

# model.add(Dense(5, input_dim = 1))
# model.add(Dense(50))
# model.add(Dense(250))
# model.add(Dense(500))
# model.add(Dense(1000))
# model.add(Dense(500))
# model.add(Dense(250))
# model.add(Dense(50))
# model.add(Dense(5))
# model.add(Dense(1)) - 이거는 왜 값이 위에 것 보다 안좋을까? 노드 개수도 많은데..

# 3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=100, batch_size=1,
          validation_data=(x_val, y_val))
'''# val을 fit에 적용.'''

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

# RMSE 수치는 0에 가깝게 R2는 1에 가깝게 해야 좋은 모델
# val 을 사용함으로써 더 좋은 모델이 만들어진다.
# 터미널창을 보면 val_loss, val_mse 등이 보인다. val을 넣었기 때문에 적용이 되기 시작했다는 것 확인 가능
'''
보면 loss와 val_loss 틀려. 왜냐 할때 마다 검증하잖아,
통상적으로 훈련(공부)할 때보다 평가(시험)했을 때 결과치(성적)이 떨어지지?
그래서 loss가 상대적으로 val_loss 구간이 더 높아
그리고 RMSE나 R2는 val구간이 좀 더 낮다.? - 이거 맞나여
'''
# 지금은 데이터의 갯수가 너무 적어. 하지만 프로젝트시 데이터가 엄청 많기 때문에 지금 연습해보고 나중에 고생을 줄이자.

# 데이터 분포그래프
'''
105|-------------------------------.
104|                           /   |
103|                       /       |
102|                  /            |
101|--------------.                |
.  |             /|                |
.  |             /|                |
.  |            / |                |
15 |-----------.  |                |
   |        /  |  |                |
   |      /    |  |                |
   |   /       |  |                |
   | /         |  |                |
ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
    1 2 3 ...  15 101 102 103 104 105
'''
# w:     1,      A,        1         => 붙어있는 데이터이기 때문에 가중치가 다르다.
