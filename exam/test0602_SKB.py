# 20-06-02 화 시험
# 06-03 아침 삼성전자 시가 맞추기

''' 시험 조건
1. 6/3 삼성전자 시가 맞추기
2. CSV 데이터는 건들지 말 것
3. 앙상블 모델 사용
   hite + samsung
4. 6시 mail 발송 제목은 "손기범 [0602시험] 57000원"
   첨부   : 소스, npy, h5
   소스명 : test0602_SKB.py
'''
# 날짜 순이 거꾸로 되어있다. 알아서 처리

### 1. 데이터 ###

# data load
import numpy as np
import pandas as pd

hite = np.load('./data/hite.npy')
samsung = np.load('./data/samsung.npy')
# print(hite)
# print(samsung)
print(hite.shape)     # (508, 5)
print(samsung.shape)  # (508, 1)

# 데이터 구성 생각 (이상한 생각인 것 같다)
'''
 DNN 앙상블로 구현할 거다

 데이터
 x1, y1
 x2, y2 로 구성해야한다.

 x1 을 hite 전체 / y1 을 hite 시가
 x2 를 hite 전체 / y2 를 samsung 시가 로 구성하자.

 두 개의 레이어에 대한 input 2개를 구성하고 병합 후, output 2개를 구성하자 예측값을 뽑아본다.

 ## 구성시 
 총 508일차 / 4 = 127
 x= 3일씩, y = 그다음 4일차만 로 자르기 
 >>> 이 생각 문제 있음 이상한데 데이터 구성같음 '''
# x1, y1 = hite 전체, hite 시가
# x2, y2 = hite 전체, samsung 시가
def split_xy3(dataset, time_steps, y_column):
   x, y = list(), list()
   for i in range(len(dataset)):
      x_end_number = i + time_steps
      y_end_number = x_end_number + y_column

      if y_end_number > len(dataset):
         break
      tmp_x = dataset[i:x_end_number, :]
      tmp_y = dataset[x_end_number:y_end_number, 0]
      x.append(tmp_x)
      y.append(tmp_y)
   return np.array(x), np.array(y)
x1, y1 = split_xy3(hite, 3, 1)
x2, y2 = split_xy3(samsung, 3, 1)

print(x1.shape) # (505, 3, 5)
print(y1.shape) # (505, 1)
print(x2.shape) # (505, 3, 1)
print(y2.shape) # (505, 1) 

# reshape
x1 = x1.reshape(x1.shape[0], x1.shape[1] * x1.shape[2])
x2 = x2.reshape(x2.shape[0], x2.shape[1] * x2.shape[2])
print(x1.shape) # (505, 15)
print(x2.shape) # (505, 3)

# 전처리 minmax
from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = StandardScaler()
x1_scale = scaler.fit_transform(x1)
x2_scale = scaler.fit_transform(x2)
print(x1_scale.shape) # (505, 15)
print(x2_scale.shape) # (505, 3)

# train_test_split
from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test = train_test_split(
    x1_scale, y1, random_state=66, shuffle = True,
    train_size = 0.8)

x2_train, x2_test, y2_train, y2_test = train_test_split(
    x2_scale, y2, random_state=66, shuffle = True,
    train_size = 0.8)

print(x1_train.shape) # (404, 15)
print(x1_test.shape)  # (101, 15)
print(y1_train.shape) # (404, 1)
print(y1_test.shape)  # (101, 1)

print(x2_train.shape) # (404, 3)
print(x2_test.shape)  # (101, 3)
print(y2_train.shape) # (404, 1)
print(y2_test.shape)  # (101, 1)

print(x1_test[-1])


### 2. 모델 ###
from keras.models import Model
from keras.layers import Dense, Input
from keras.layers import Dropout

# Input 1 #
input1 = Input(shape=(15,))

dense1_1 = Dense(100, name='dense1_1')(input1)
dense1_1 = Dense(200, )(dense1_1)
dense1_1 = Dense(500, )(dense1_1)
dp1_1 = Dropout(0.2)(dense1_1)
dense1_1 = Dense(200, )(dp1_1)
dense1_1 = Dense(50, )(dense1_1)

# Input 2 #
input2 = Input(shape=(3,))

dense2_1 = Dense(100, name='dense2_1')(input2)
dense2_1 = Dense(200, )(dense2_1)
dense2_1 = Dense(500, )(dense2_1)
dp2_1 = Dropout(0.2)(dense2_1)
dense2_1 = Dense(200, )(dp2_1)
dense2_1 = Dense(50, )(dense2_1)

# 병합 #
from keras.layers.merge import concatenate
merge1 = concatenate([dense1_1, dense2_1])

middle1 = Dense(300, name='mid1')(merge1)
middle1 = Dense(150)(middle1)
output = Dense(1)(middle1)

# # output 1 #
# output1 = Dense(30)(middle1)
# output1_2 = Dense(15)(output1)
# output1_3 = Dense(1, name='out1_3')(output1_2)

# # output 2 #
# output2 = Dense(25)(middle1)
# output2_2 = Dense(10)(output2)
# output2_3 = Dense(1, name='out2_3')(output2_2)

# 모델 명시 #
model = Model(inputs=[input1, input2],
              outputs=output)
model.summary()


### 3. 훈련
model.compile(optimizer='adam', loss='mse', metrics=['mse'])

##### EarlyStopping & Modelcheckpoint & Tensorboard
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
es = EarlyStopping(monitor='val_loss', patience=50, mode='auto')

modelpath = './exam/test-{epoch:02d}-{val_loss:.4f}.hdf5'
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss',
                     save_best_only=True, save_weights_only=False)

tb = TensorBoard(log_dir='graph', histogram_freq=0,
                 write_graph=True, write_images=True)

model.fit([x1_train, x2_train], y2_train,
          epochs=300, batch_size=32, verbose=2,
          validation_split=0.25,
          callbacks=[es])

# """ model 저장 """
# model.save('./exam/test0602_SKB.h5')


### 4. 평가, 예측
loss, mse = model.evaluate([x1_test, x2_test], y2_test,
                           batch_size=32)
print('mse', mse)

### 3번
y2_pred = model.predict([x1_test, x2_test])

for i in range(1):
   print('시가 : ', y2_test[i], '/ 예측가 : ', y2_pred[i])
