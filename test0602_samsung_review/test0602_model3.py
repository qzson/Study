# 20-06-03
# input 2개 lstm으로 구성하기

import numpy as np
from keras.models import Model, load_model
from keras.layers import Dense, LSTM, Dropout, Input
from keras.layers.merge import concatenate, Concatenate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

# cnn 할 때, 더 결과가 좋은 경우가 있다. conv1d 와 연관?
# LSTM 2개 구현 구성하라

def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1):
        subset = seq[i:(i+size)]
        aaa.append([j for j in subset])
    # print(type(aaa))
    return np.array(aaa)

size = 6            # 6일치씩 자르겠다?

### ==== 1. 데이터
# npy 불러오기
samsung = np.load('./data/samsung_test.npy', allow_pickle='True')
hite = np.load('./data/hite_test.npy', allow_pickle='True')

print(samsung.shape) # (509, 1)
print(hite.shape)    # (509, 5)

samsung = samsung.reshape(samsung.shape[0],) # (509, )

samsung = (split_x(samsung, size))
print(samsung.shape) # (504, 6, 1) / # (504, 6)
print(samsung)

# 삼성만 x,y를 만들어주고 하이트는 y가 필요없다.
x_sam = samsung[:, 0:5]
y_sam = samsung[:, 5]

print(samsung[0:5])
print(x_sam[0])
print(y_sam[0])
print(x_sam.shape) # (504, 5)
print(y_sam.shape) # (504, )

x_hit = hite[5:510, :]
print(x_hit)
print(x_hit.shape)     # (504, 5)

# LSTM 구성 위해서 리쉐이프
x_sam = x_sam.reshape(x_sam.shape[0],5,1) # (504, 5, 1)
x_hit = x_hit.reshape(x_hit.shape[0],5,1)

### ==== 2. 모델

input1 = Input(shape=(5,1))
x1 = LSTM(10)(input1)
x1 = Dense(10)(x1)

input2 = Input(shape=(5,1))
x2 = LSTM(5)(input2)
x2 = Dense(5)(x2)

merge = Concatenate()([x1, x2])

output = Dense(1)(merge)

model = Model(inputs=[input1, input2], outputs=output)

model.summary()


### ==== 3. 실행, 훈련
model.compile(optimizer='adam', loss='mse')
model.fit([x_sam, x_hit], y_sam, epochs=5)
# 앙상블 시, 행까지 맞춰줘야 한다.
