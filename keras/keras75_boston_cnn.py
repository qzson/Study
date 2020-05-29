# 20-05-29 / 1425 ~
# data - > CSV - > pandas or numpy - > numpy -> .npy 로 저장 (중요) - > DB 에 저장
# pandas : 자료형이 섞여있을 때
# numpy  : 자료형이 1개일 때


import numpy as np
from sklearn.datasets import load_boston

dataset = load_boston()
x = dataset.data
y = dataset.target

print(x.shape) # (506, 13)
print(y.shape) # (506,)

###
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(x)
x_scaled = scaler.transform(x)
print(x_scaled)

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca.fit(x_scaled)
x_pca = pca.transform(x_scaled)
print(x_pca)


###
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x_pca, y, random_state=66, shuffle=True,
    train_size = 0.8)

print(x_train.shape) #(404, 2)
print(x_test.shape)  #(102, 2)
print(y_train.shape) #(404,)
print(y_test.shape)  #(102,)

x_train = x_train.reshape(404, 2, 1, 1)
x_test = x_test.reshape(102, 2, 1, 1)

### 2. 모델
from keras.models import Sequential
from keras.layers import Dense, LSTM, Conv2D
from keras.layers import Flatten, MaxPooling2D, Dropout

model = Sequential()

model.add(Conv2D(30, (1,1), input_shape=(2,1,1)))
model.add(Flatten())
model.add(Dense(1))
model.add(Dense(1))
model.add(Dense(1))

model.summary()

# EarlyStopping
from keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor = 'loss', patience=100, mode = 'auto')

modelpath = './model/{epoch:02d}-{val_loss:.4f}.hdf5'
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss',
                     save_best_only=True, mode='auto')

### 3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train,
          epochs=300, batch_size=32, verbose=1,
          validation_split=0.25,
          callbacks=[es, cp])


### 4. 평가, 예측
loss, mse = model.evaluate(x_test, y_test, batch_size=32)
print('loss:', loss)
print('mse:', mse)

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

''' ㅅㅂ
RMSE :  6.511550343917062
R2 :  0.49271598684009066
'''