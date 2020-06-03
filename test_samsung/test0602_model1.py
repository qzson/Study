# 20-06-03
# 그냥 dnn

import numpy as np
from keras.models import Model, load_model
from keras.layers import Dense, LSTM, Dropout, Input
from keras.layers.merge import concatenate, Concatenate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

# cnn 할 때, 더 결과가 좋은 경우가 있다. conv1d 와 연관?

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
print(samsung.shape) # 스칼라로 리쉐이프 전 : (504, 6, 1) / 리쉐이프 후 : # (504, 6)

# 삼성만 x,y를 만들어주고 하이트는 y가 필요없다.
x_sam = samsung[:, 0:5]
y_sam = samsung[:, 5]

print(samsung[0])
print(x_sam[0])
print(y_sam[0])
print(x_sam.shape) # (504, 5)
print(y_sam.shape) # (504, )

x_hit = hite[5:510, :]
print(x_hit.shape)     # (504, 5)

### ==== 2. 모델

input1 = Input(shape=(5,))
x1 = Dense(50)(input1)
x1 = Dense(1000)(x1)
drop1 = Dropout(0.2)(x1)
x1 = Dense(500)(drop1)

input2 = Input(shape=(5,))
x2 = Dense(50)(input2)
x2 = Dense(1000)(x2)
drop2 = Dropout(0.2)(x2)
x2 = Dense(500)(drop2)

merge = concatenate([x1, x2])

output = Dense(1)(merge)

model = Model(inputs=[input1, input2], outputs=output)

model.summary()


### ==== 3. 실행, 훈련
model.compile(optimizer='adam', loss='mse')
model.fit([x_sam, x_hit], y_sam, epochs=300, batch_size=32, verbose=1)
# 앙상블 시, 행까지 맞춰줘야 한다.


### ==== 4. 평가, 예측
mse = model.evaluate([x_sam, x_hit], y_sam,
                           batch_size=32)
print('mse', mse)

y_sam = model.predict([x_sam, x_hit])

for i in range(5):
   print('시가 : ', x_sam[i], '/ 예측가 : ', y_sam[i])