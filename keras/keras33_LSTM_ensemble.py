# 20.05.21
# 앙상블 모델을 만드시오.

'''
튜닝 결과값
loss: 9.9162e-04 [[82.63302]]
loss: 1.9102e-04 [[83.06454]]
loss: 3.4185e-05 [[80.8554]]
loss: 3.1168e-05 [[82.878944]]
loss: 1.7358e-06 [[83.3059]]
loss: 1.0758e-07 [[83.19377]]
loss: 3.6337e-09 [[83.03]]
loss: 6.4642e-10 [[83.497444]]
결론 = 80 ~ 84 까지의 값이 좋은 값이다 ?
'''

from numpy import array
from keras.models import Model
from keras.layers import Dense, LSTM, Input


# 1. 데이터
x1 = array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
           [5,6,7],[6,7,8],[7,8,9],[8,9,10],
           [9,10,11],[10,11,12],
           [20,30,40],[30,40,50],[40,50,60]])
x2 = array([[10,20,30],[20,30,40],[30,40,50],[40,50,60],
           [50,60,70],[60,70,80],[70,80,90],[80,90,100],
           [90,100,110],[100,110,120],
           [2,3,4],[3,4,5],[4,5,6]])
y = array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x1_predict = array([55,65,75])
x2_predict = array([65,75,85])


print("x1.shape:", x1.shape) # (13, 3)
print("x2.shape:", x2.shape) # (13, 3)
print("y.shape:", y.shape) # (13,)

# x = x.reshape(4, 3, 1)                      
x1 = x1.reshape(x1.shape[0], x1.shape[1], 1) # (13, 3, 1)
x2 = x2.reshape(x2.shape[0], x2.shape[1], 1) # (13, 3, 1)

print(x1.shape)
print(x2.shape)


# 2. 모델구성

######### 모델 1 #########
input1 = Input(shape =(3,1))
dense1_1 = LSTM(60, activation='relu', name='dense1_1')(input1)
dense1_1 = Dense(90, activation='relu', name='dense1_2')(dense1_1)
dense1_1 = Dense(120, activation='relu', name='dense1_3')(dense1_1)
dense1_2 = Dense(150, activation = 'relu')(dense1_1)
   

######### 모델 2 #########
input2 = Input(shape =(3,1)) 
dense2_1 = LSTM(60, activation='relu', name='dense2_1')(input2)
dense2_1 = Dense(90, activation='relu', name='dense2_2')(dense2_1)
dense2_1 = Dense(120, activation='relu', name='dense2_3')(dense2_1)
dense2_2 = Dense(150, activation = 'relu')(dense2_1)
  

######### 모델 병합#########
from keras.layers.merge import concatenate   
merge1 = concatenate([dense1_2, dense2_2])   

middle1 = Dense(15)(merge1)
middle1 = Dense(20)(middle1) 
middle1 = Dense(25)(middle1) 
middle1 = Dense(30)(middle1) 
middle1 = Dense(35)(middle1) 


######### output 모델 구성 ###########
output1 = Dense(35)(middle1)   
output1_2 = Dense(40)(output1)
output1_2 = Dense(50)(output1_2)
output1_2 = Dense(30)(output1_2)
output1_2 = Dense(20)(output1_2)
output1_2 = Dense(10)(output1_2)
output1_2 = Dense(5)(output1_2)
output1_3 = Dense(1)(output1_2) 


######### 모델 명시 #########
model = Model(inputs = [input1, input2],
              outputs= output1_3) 

model.summary()

# 3. 실행
model.compile(optimizer='adam', loss='mse')
model.fit([x1, x2], y, epochs=800, batch_size=32, verbose=2)

x1_predict = x1_predict.reshape(1,3,1)
x2_predict = x2_predict.reshape(1,3,1)


# 4. 예측
# print(x1_predict)
# print(x2_predict)
y_predict = model.predict([x1_predict, x2_predict])
print(y_predict)