# 20.05.22_10day_1000~
# 함수형으로 리뉴얼하시오.
# Dense로 모델 구성하시오 (LSTM다 치우고)

'''
튜닝 값
LSTM vs Dense 기본 모델
LSTM 모델에 비해서 적은 노드의 수로도 값이 상대적으로 잘 나온다.
84.862755
loss: 3.1056e-04 // [[85.08172]]
왜 그럴까 ?
아마, 갖고 있는 데이터 양이 적어서 그러지 않을까?
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
x_predict = array([55,65,75]) # (3,)

print("x.shape:", x.shape) # (13, 3)
print("y.shape:", y.shape) # (13,)

# x = x.reshape(13, 3, 1)                      
# x = x.reshape(x.shape[0], x.shape[1], 1) # (13, 3, 1)
print(x.shape)


# 2. 모델구성
input1 = Input(shape=(3,))
dense1 = Dense(10, activation='relu', name='dense1_1',)(input1)
dense1 = Dense(20, activation='relu', name='dense1_2')(dense1)
dense1 = Dense(50, activation='relu', name='dense1_3')(dense1)
dense1 = Dense(20, activation='relu', name='dense1_4')(dense1)
dense1 = Dense(10, activation='relu', name='dense1_5')(dense1)
dense1 = Dense(5, activation='relu', name='dense1_6')(dense1)
output1 = Dense(1, name='output1')(dense1)

model = Model(inputs=input1, outputs=output1)

model.summary()


# 3. 훈련
model.compile(optimizer='adam', loss='mse')

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=100, mode='auto')

model.fit(x, y, epochs=450, batch_size=5, verbose=2,
          callbacks=[early_stopping])

x_predict = x_predict.reshape(1,3)
# print(x_predict)
# (3 , )  = [[55]            벡터       input :  x = [[1, 2, 3]
#            [65]                      (13, 3)        [2, 3, 4]
#            [75]]                                    [3, 4, 5]]  
# (1, 3 ) = [[55, 65, 75]]

# 4. 예측
y_predict = model.predict(x_predict)
print(y_predict)

