# 200518 0900 ~
# r2 (결정계수) 0.95와 acc 0.95와 엄밀히 같지는 않지만, 비슷한 성능? 결과치라고 예상은 할 수 있다.
# rmse, rma, r2와 같이 보조지표를 사용한다.
# mlp 멀티 레이어 퍼셉트론 / 여러가지 데이터 셋 들어가는 것 했었다. input, output 2 이상인 것들 했었다.
# train, val, test 셋 에 대해서 분리를 했었다. 항상 이렇게 나눠서 하는 작업을 해야한다.
# 매트릭스 부분 acc넣고 테스트 했었고 mse로 바꿨다. 그 보는 부분 역시 딜레이가 있어서 vervose를 사용했었다.

'''
1. Sequential 모델
2. 함수형
                o (input)
    o                         o
ooooooooo                oooooooooo
ooooooooo                oooooooooo
ooooooooo                oooooooooo
ooooooooo                oooooooooo
    o                         o
                o (output)
이렇게 모델 2개를 엮어서 함수형 2개를 사용 가능

''' 

# 1. 데이터
import numpy as np
x = np.array([range(1,101), range(311, 411), range(100)])
y = np.array(range(711,811))

x = np.transpose(x)

print(x.shape)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle = False,
    train_size = 0.8)

print(x_train)
print(x_test)


# 2. 모델구성
from keras.models import Sequential, Model # 대문자 시작으로 모델 쓰면 함수형 모델을 땡겨쓴다는 의미
from keras.layers import Dense, Input # 함수형에서는 인풋에 대해서 명시 해줘야한다.
# model = Sequential()
# model.add(Dense(5, input_dim = 3))
# model.add(Dense(4))
# model.add(Dense(1))

# 함수형 모델은 인풋이 무엇인지 아웃풋이 무엇인지 명시를 해야한다.
# 함수형 모델 사용 시 shape를 사용한다. 행을 뺀 나머지지 부분 나열.
# 첫번째 인풋레이어 구성이 된다.
# 함수형은 각 레이어별로 이름을 지정해줘야한다.
input1 = Input(shape=(3,)) # 변수명은 소문자로 해주는다는 것 (input1 말고도 아무거나 상관없다.)
# 케라스의 레이어스 라는 계층의 친구다
dense1 = Dense(5, activation='relu')(input1) #activation - 활성화 / 함수 꽁다리에 변수명 명시해준다.
dense1 = Dense(25, activation='relu')(dense1)
dense1 = Dense(50, activation='relu')(dense1)
dense1 = Dense(250, activation='relu')(dense1)
dense1 = Dense(500, activation='relu')(dense1)
dense1 = Dense(250, activation='relu')(dense1)
dense1 = Dense(50, activation='relu')(dense1)
dense1 = Dense(25, activation='relu')(dense1)
dense1 = Dense(5, activation='relu')(dense1)
output1 = Dense(1)(dense1) # y 열에 따라서 값 수정

# dense1로 통일해줘도 어차피 머신에서는 dense_1..2...3... 이렇게 잡하서 해준다.
# dense1, 2 ,3 이렇게 굳이 할 필요도 없다. 하지만 쓰는 것이 좋다 ?
# 앙상블때 또 나온다

model = Model(inputs=input1, outputs=output1)
# 시퀀셜 모델에서는 시퀀셜이라고 정의해줬다. ex) model = Sequential()
# 이것도 마찬가지로 다르게 정의.

model.summary()

# 시퀀셜 모델은 2개를 엮을 수 없다.
# 2개의 모델이라는 것은 거대한 데이터 셋이 있을 수 있다?>
# 각 데이터별로 모델을 만들어서 합친다 ? 가능하다. => 그렇게 하는 것을 : 앙상블 이라고 말한다.
# 앙상블 모델이려면 2개 이상의 모델이 되어야하고 2개 이상의 데이터 셋이 있어야 된다.
# 1개의 데이터 셋이라면 시퀀셜 쓰면 되는데, 2개면 각자의 모델을 훈련 시키고 합치는 앙상블을 쓴다.

# 3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=200, batch_size=1,
            verbose=2,
          validation_split=0.25)

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