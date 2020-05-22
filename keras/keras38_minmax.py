# 20.05.22_10day_1425~
# 데이터 전처리 중 MINMAX
# 당연히 하는 것

# minmaxscaler(정규화)
'''
- 데이터를 0-1 사이 값으로 변환
- (x-x의 최소값) / (x의 최대값 - x의 최소값)
- 데이터의 최소, 최대 값을 알 경우 사용
'''
# standardscaler(표준화)
'''
- 기존 변수에 범위를 정규 분포로 변환
- (x-x의 평균값) / (x의 표준편차)
- 데이터의 최소, 최대 값을 모를 경우 사용
'''
# Robust Scaler
'''
- 잘 사용 X
- 중앙값이 0, IQR이 1이 되도록 변환
- StandardScaler에 의한 표준화보다 동일한 값을 더 넓게 분포
- 이상치를 포함하는 데이터를 표준화하는 경우
'''
# MaxAbsScaler
'''
- 0을 기준으로 절대값이 가장 큰 수가 1 또는 -1이 되도록 변환
'''

from numpy import array
from keras.models import Model
from keras.layers import Dense, LSTM, Input


# 1. 데이터
x = array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
           [5,6,7],[6,7,8],[7,8,9],[8,9,10],
           [9,10,11],[10,11,12],
           [2000,3000,4000],[3000,4000,5000],[4000,5000,6000], # (14,3)
           [100,200,300]])
y = array([4,5,6,7,8,9,10,11,12,13,5000,6000,7000,400])
x_predict = array([55,65,75]) # (3,)

x_predict = x_predict.reshape(1,3)


from sklearn.preprocessing import MinMaxScaler, StandardScaler
# scaler = MinMaxScaler()
scaler = StandardScaler()
scaler.fit(x)                             # 전처리 실행 : x를 넣어서 MinMax_scaler, Standard_scaler 실행하겠다.
                                          #              scaler에 실행한 값 저장 ( x의 범위 )
x = scaler.transform(x)                   # x의 모양을 MinMaxScaler을 실행한 값으로 바꿔주겠다.
x_predict = scaler.transform(x_predict)   # x의 범위로 계산한 sclar 값에서 x_predict에 해당되는 값을 가져오겠다.
print(x) # 0~1사이로 깔끔하게 압축이 되었다.
print(x_predict)


print("x.shape:", x.shape) # (14, 3)
print("y.shape:", y.shape) # (14,)

# x = x.reshape(14, 3, 1)                      
x = x.reshape(x.shape[0], x.shape[1], 1) # (14, 3, 1)

print(x.shape)


# 2. 모델구성
input1 = Input(shape=(3,1))

LSTM1 = LSTM(1000, return_sequences= True)(input1)
# LSTM2 = LSTM(10)(LSTM1, return_sequences= True)(LSTM1)  # return_sequences를 썼으면 무조건 LSTM사용
LSTM2 = LSTM(500)(LSTM1)           
dense1 = Dense(10)(LSTM2)        
dense2 = Dense(10)(dense1)     
dense2 = Dense(10)(dense2)   
dense2 = Dense(10)(dense2)                     
dense2 = Dense(10)(dense2)                     
dense2 = Dense(10)(dense2)                     
dense3 = Dense(10)(dense2)

output1 = Dense(1, name='output1')(dense3)

model = Model(inputs=input1, outputs=output1)

model.summary()

# 3. 훈련
model.compile(optimizer='adam', loss='mse')

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=100, mode='auto')

model.fit(x, y, epochs=800, batch_size=16, verbose=2,
          callbacks=[early_stopping])

x_predict = x_predict.reshape(1,3,1)
print(x_predict)

# 4. 예측
y_predict = model.predict(x_predict)
print(y_predict)