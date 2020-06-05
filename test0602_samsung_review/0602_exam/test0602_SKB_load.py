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


# """ model 저장 """
# model.save('./exam/test0602_SKB.h5')
from keras.models import load_model
model = load_model('./test0602_samsung_review/0602_exam/test0602_SKB.h5')

### 4. 평가, 예측
loss, mse = model.evaluate([x1_test, x2_test], y2_test,
                           batch_size=32)
print('mse', mse)

### 3번
y2_pred = model.predict([x1_test, x2_test])

for i in range(3):
   print('시가 : ', y2_test[i], '/ 예측가 : ', y2_pred[i])
