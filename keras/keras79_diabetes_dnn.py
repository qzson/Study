# 20-05-30 / 토요일

# 당뇨병 DNN 모델 구성
# 이건 2진 분류 모델이 아니라, 회귀 모델로 생각된다. 그러므로, 분류 모델 안쓸 것임.


### 1. 데이터
import numpy as np
from sklearn.datasets import load_diabetes

x, y = load_diabetes(return_X_y=True)

# 데이터 load 선생님 방식. 근데 이상한 에러떠서 보류
# dataset = load_diabetes()
# x = dataset.data
# y = dataset.target
# print(x)

print(x.shape) # (442, 10)
print(y.shape) # (442, )


### 데이터 전처리
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# scaler = StandardScaler()
# scaler.fit(x)
# x_scaled = scaler.transform(x)

sc = MinMaxScaler()
sc.fit(x)
x = sc.transform(x)

# from sklearn.decomposition import PCA

# pca = PCA(n_components=2)
# pca.fit(x_scaled)
# x_pca = pca.transform(x_scaled)
# print(x_pca.shape)   # (442, 2)
 ## 짚고 넘어가야할 점
 # 여기서 데이터 전처리로 PCA를 왜 써야하는지 에 대한 궁금사항이 있다.
 # 사실 이번 모델에서도 왜 쓰는 것인지 스스로 인지하고 있지 못함


### Train_Test_Split
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=66, shuffle=True,
    train_size = 0.8)
print(x_train.shape)  # (353, 2) : pca시 // (353, 10)
print(x_test.shape)   # (89, 2)  : pca시 // (89, 10)
print(y_train.shape)  # (353, )
print(y_test.shape)   # (89, )


#### 2. 모델
from keras.models import Sequential
from keras.layers import Dense, Dropout

model = Sequential()

model.add(Dense(300, activation='linear', input_shape=(10,)))
model.add(Dropout(0.2))
model.add(Dense(400, activation='linear'))
model.add(Dense(400, activation='linear'))
model.add(Dropout(0.3))
model.add(Dense(200, activation='linear'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='linear'))

model.summary()


# EarlyStopping & ModelCheckpoint(use x) & Tensorboard(use x)
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
es = EarlyStopping(monitor='val_loss', patience=50, mode='auto')

modelpath = './model/{epoch:02d}-{val_loss:.4f}.hdf5'
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss',
                     save_best_only=True, mode='auto')

tb = TensorBoard(log_dir='graph', histogram_freq=0,
                 write_graph=True, write_images=True)


### 3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train,
          epochs=100, batch_size=32, verbose=2,
          validation_split=0.3,
          callbacks=[es])
 

### 4. 평가, 예측
loss, mse = model.evaluate(x_test, y_test, batch_size=32)
print("loss :", loss)
print("mse :", mse)

y_pred = model.predict(x_test)
# print(y_pred)

# RMSE
from sklearn.metrics import mean_squared_error, r2_score
def RMSE(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE : ", RMSE(y_test, y_pred))

# R2
r2 = r2_score(y_test, y_pred)
print("R2 : ", r2)

'''
activation = linear
mse : 3300.040771484375
RMSE :  57.44598025754967
R2 :  0.4915223257639485

activation = relu 사용 시, R2 : 0.2
'''