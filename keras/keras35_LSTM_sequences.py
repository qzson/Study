# 20.05.22_10day_1000~
# return_sequences

'''
튜닝 값
loss: 1.1295e-04 [[79.91169]]
loss: 1.8396e-04 [[80.3458]]
'''

from numpy import array
from keras.models import Model
from keras.layers import Dense, LSTM, Input


# 1. 데이터
x = array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
           [5,6,7],[6,7,8],[7,8,9],[8,9,10],
           [9,10,11],[10,11,12],
           [20,30,40],[30,40,50],[40,50,60]])
y = array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x_predict = array([50,60,70]) # (3,)

print("x.shape:", x.shape) # (13, 3)
print("y.shape:", y.shape) # (13,)

# x = x.reshape(13, 3, 1)                      
x = x.reshape(x.shape[0], x.shape[1], 1) # (13, 3, 1)
print(x.shape)


# 2. 모델구성
input1 = Input(shape=(3,1))
LSTM1 = LSTM(200, activation='relu', return_sequences=True)(input1)
LSTM2 = LSTM(100, activation='relu')(LSTM1)
dense1 = Dense(50, activation='relu')(LSTM2)
dense1 = Dense(25, activation='relu')(dense1)
dense1 = Dense(15, activation='relu')(dense1)
output1 = Dense(1, name='output1')(dense1)

model = Model(inputs=input1, outputs=output1)

model.summary()
'''
# <ValueError: Input 0 is incompatible with layer dense1_2: expected ndim=3, found ndim=2>
# LSTM 의 형태는 x=(행,열,몇)=(batch,time,feature) : 3차원
# Dense 레이어 자체는 2차원(행,열)만 입력 받음
# 윗 레이어의 아웃풋이 2차원, 근데 LSTM이라 3차원을 받아야함
# 즉, 맨위 오류의 의미는 "3차원으로 바꿔라" 라는 의미.
# return_sequences 을 사용하면 차원을 유지시켜준다.
# Dense 모델은 2차원을 받아들이기 때문에 상위레이어가 LSTM인 지점에 return_sequences를 사용하면 역시나 오류가 뜬다.

LSTM  = (  ,  ,  ) : 3 차원
Dense = (  ,  )    : 2 차원


Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         (None, 3, 1)              0
_________________________________________________________________
lstm_1 (LSTM)                (None, 3, 10)             480
_________________________________________________________________
lstm_2 (LSTM)                (None, 10)                840
_________________________________________________________________
dense_1 (Dense)              (None, 5)                 55
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 6
=================================================================
# 앞에 output_node가 input_dim(feature)가 된다.
# LSTM_sequences_parameter
 :num_param = 4 * (  num_units   +   input_dim  + bias) * num_units
            = 4 * (LSTM2_output  + LSTM1_output +   1 ) * LSTM2_output
            = 4 * (    10    +    10     +   1 ) * 10
            = 840

or another 공식
 :num_param = 4 * [(size_of_input  + bias) * size_of_output + size_of_output^2]
            = 4 * [(LSTM1_output +   1 ) * LSTM2_output + LSTM2_output^2]
            = 4 * [(    10       +   1 ) *      10      +       100     ]
            = 840
# [(10 + 1) * 10 + 100] * 4 = 840

input_dim이 10인 이유

내 정리
    h(t-1), h(t), h(t2)... h(t9) = 10
(lstm1) o o o o o o o o o o

    h(t-1), h(t), h(t2)... h(t9) = 10
-> input_dim 으로 연산 = 10
(lstm2) o o o o o o o o o o

샘 정리
= 아웃풋 노드의 개수가 feature로 들어간다.
'''

# 3. 훈련
model.compile(optimizer='adam', loss='mse')

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=100, mode='auto')

model.fit(x, y, epochs=500, batch_size=16, verbose=2,
          callbacks=[early_stopping])

x_predict = x_predict.reshape(1,3,1)
print(x_predict)

# 4. 예측
y_predict = model.predict(x_predict)
print(y_predict)