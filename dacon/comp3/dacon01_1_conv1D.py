# 20-06-09
# Dacon : 진동데이터 활용 충돌체 탐지
# 진동데이터로 충돌체의 x좌표, y좌표, 질량, 속도 예측

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

train_x = pd.read_csv('./data/dacon/comp3/train_features.csv', header=0, index_col=0)
train_y = pd.read_csv('./data/dacon/comp3/train_target.csv', header=0, index_col=0)
test_x = pd.read_csv('./data/dacon/comp3/test_features.csv', header=0, index_col=0)


from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Dropout, Input, Conv1D, Flatten, MaxPooling1D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA

# train_x_tmp = train_x.iloc[:375,:]
# train_x_tmp2 = train_x.iloc[375:750,:]
# # train_x_tmp = train_x.iloc[:375,:]
# # train_x_tmp = train_x.iloc[:375,:]
# print(train_x_tmp.tail(5))  # id : 0 , (0  ~374행의 데이터 : 375개)
# #         Time        S1        S2         S3          S4
# # id
# # 0   0.001480 -64168.90 -64168.90   52279.59  106792.600
# # 0   0.001484 -64236.79 -64236.79   16518.64   58248.420
# # 0   0.001488 -63755.95 -63755.95  -25270.30    3015.649
# # 0   0.001492 -63020.44 -63020.44  -65904.66  -49795.140
# # 0   0.001496 -61808.07 -61808.07 -102329.20  -95687.360

# print(train_x_tmp2.tail(5)) # id : 1 , (375~749행의 데이터 : 375개)
# #         Time         S1        S2         S3         S4
# # id
# # 1   0.001480   962514.8  318184.5  396361.80 -641504.60
# # 1   0.001484  1031978.0  354831.0  248709.80 -528058.80
# # 1   0.001488  1069688.0  354107.2   93556.34 -394603.00
# # 1   0.001492  1068054.0  310828.9  -42801.14 -246912.00
# # 1   0.001496  1020689.0  230333.6 -156492.70  -75997.95


# print(train_x.head(5))
print(train_x.shape)    # (1050000, 5)
print(train_y.shape)    # (2800, 4)
print(test_x.shape)     # (262500, 5)

x = train_x.iloc[:, -4:]
y = train_y
print(x.shape)          # (1050000, 4)
print(y.shape)          # (2800, 4)

x_pred = test_x.iloc[:, -4:]
print(x_pred.shape)     # (262500, 4)

# npy 형변환
x = x.values
y = y.values
x_pred = x_pred.values

x = x.reshape(2800, 375, 4)
x_pred = x_pred.reshape(700, 375, 4)
print(x.shape)      # (2800, 375, 4)
print(x_pred.shape) # (700, 375, 4)

# 스플릿
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=66, shuffle=True)
print(x_train.shape)    # (2240, 375, 4)
print(x_test.shape)     # (560, 375, 4)
print(y_train.shape)    # (2240, 4)
print(y_test.shape)     # (560, 4)

x_train = x_train.reshape(840000, 4)
x_test = x_test.reshape(210000, 4)
x_pred = x_pred.reshape(262500, 4)
# print(x_train.shape)    # (2240, 375, 4)
# print(x_test.shape)     # (560, 375, 4)

# 전처리
sc = MinMaxScaler()
sc.fit(x_train)
x_train = sc.transform(x_train)
x_test = sc.transform(x_test)
x_pred = sc.transform(x_pred)
print(x_train[0])

x_train = x_train.reshape(2240, 375, 4)
x_test = x_test.reshape(560, 375, 4)
x_pred = x_pred.reshape(700, 375, 4)


''' 2. 모델 '''
model = Sequential()
model.add(Conv1D(150, 2, input_shape=(375, 4)))
# model.add(MaxPooling1D())
model.add(Conv1D(300, 2, padding='same'))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(50))
model.add(Dense(200)) 
model.add(Dense(50))
model.add(Dropout(0.2))
model.add(Dense(4))

model.summary()


# EarlyStopping
from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor = 'loss', patience=100, mode = 'auto')

# 3. 훈련
model.compile(loss='mse', optimizer='adam', metrics = ['mse'] )
model.fit(x_train, y_train, epochs=300, batch_size=32, verbose=1,
          validation_split=0.2,     # train의 20%
          shuffle=True,             # 셔플 사용 가능
          callbacks=[es])


# 4. 평가, 예측
loss, mse = model.evaluate(x_test, y_test, batch_size=32)
print('loss:', loss)
print('mse:', mse)

y_pred = model.predict(x_pred)
# print(y_pred)
y_pred1 = model.predict(x_test)


def kaeri_metric(y_test, y_pred1):
    '''
    y_true: dataframe with true values of X,Y,M,V
    y_pred: dataframe with pred values of X,Y,M,V
    
    return: KAERI metric
    '''
    
    return 0.5 * E1(y_test, y_pred1) + 0.5 * E2(y_test, y_pred1)


### E1과 E2는 아래에 정의됨 ###

def E1(y_test, y_pred1):
    '''
    y_true: dataframe with true values of X,Y,M,V
    y_pred: dataframe with pred values of X,Y,M,V
    
    return: distance error normalized with 2e+04
    '''
    
    _t, _p = np.array(y_test)[:,:2], np.array(y_pred1)[:,:2]
    
    return np.mean(np.sum(np.square(_t - _p), axis = 1) / 2e+04)


def E2(y_test, y_pred1):
    '''
    y_true: dataframe with true values of X,Y,M,V
    y_pred: dataframe with pred values of X,Y,M,V
    
    return: sum of mass and velocity's mean squared percentage error
    '''
    
    _t, _p = np.array(y_test)[:,2:], np.array(y_pred1)[:,2:]
    
    
    return np.mean(np.sum(np.square((_t - _p) / (_t + 1e-06)), axis = 1))

print(kaeri_metric(y_test, y_pred1))
print(E1(y_test, y_pred1))
print(E2(y_test, y_pred1))

a = np.arange(2800, 3500)
submission = pd.DataFrame(y_pred, a)
submission.to_csv('./dacon/comp3/comp3_sub2.csv', index = True, index_label= ['id'], header = ['X', 'Y', 'M', 'V'])


# submissions = pd.DataFrame({
#     "id": range(2800,3500),
#     "X": y_pred[:,0],
#     "Y": y_pred[:,1],
#     "M": y_pred[:,2],
#     "V": y_pred[:,3]
# })

# submissions.to_csv('./data/dacon/comp3/comp3_sub.csv', index = False)


# loss: 15363.877232142857
# mse: 15363.876953125
# [[-3.30301933e+01 -1.05894623e+02  2.96395245e+01  4.57785726e-01]
#  [ 4.42084084e+01 -3.02767578e+02 -1.60641422e+01  2.91339159e-01]
#  [-7.68945541e+01  1.17117836e+02  8.66010056e+01  6.02187395e-01]
#  ...
#  [ 1.33392899e+02  3.23600960e+01  1.03144440e+02  4.08849478e-01]
#  [-1.06649857e+02 -1.05677040e+02  7.45859604e+01  3.81253958e-01]
#  [-6.08396950e+01  5.41998978e+01  1.28623306e+02  5.60592890e-01]]

# loss: 15209.04296875
# mse: 15209.04296875
# 2.4048251598570514
# 2.994327866122608
# 1.815322453591495






