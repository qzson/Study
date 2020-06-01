# 20-06-01 / 월 / save 용 main 파일
# boston.sequential.dnn

''' < 코드 구성 목록>
 1. 'dataset'_model_save.h5
 2. 'dataset'_save_weights.h5
 3. 'dataset'_checkpoint_best.h5
'''

### 1. 데이터
# 데이터 불러오기
import numpy as np
from sklearn.datasets import load_boston

x, y = load_boston(return_X_y=True)
print(x.shape) # (506, 13)
print(y.shape) # (506,)

# 전처리 1. PCA
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(x)
x_scaled = scaler.transform(x)
print(x_scaled)

from sklearn.decomposition import PCA

pca = PCA(n_components=5)
pca.fit(x_scaled)
x_pca = pca.transform(x_scaled)
print(x_pca)
print(x_pca.shape)

# 전처리 2. Train_Test_Split
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x_pca, y, random_state=66, shuffle=True,
    train_size = 0.8)
print(x_train.shape) #(404, 5)
print(x_test.shape)  #(102, 5)
print(y_train.shape) #(404,)
print(y_test.shape)  #(102,)


### 2. 모델
from keras.models import Sequential
from keras.layers import Dense, Dropout

model = Sequential()

model.add(Dense(100, input_shape= (5, )))
model.add(Dense(200))
model.add(Dense(300))
model.add(Dense(400))
model.add(Dropout(0.2))
model.add(Dense(500))
model.add(Dense(400))
model.add(Dense(300))
model.add(Dense(200))
model.add(Dropout(0.2))
model.add(Dense(100))
model.add(Dense(1))

model.summary()


### 3. 훈련
# earlystopping
from keras.callbacks import EarlyStopping, ModelCheckpoint

es = EarlyStopping(monitor='val_loss', patience=50, mode='auto')

# modelcheckpoint (cp_best 값 추출 할 것)
modelpath = './model/sample/boston/boston-{epoch:02d}-{val_loss:.4f}.hdf5'
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss',
                     verbose=1,
                     save_best_only=True, save_weights_only=False)

model.compile(optimizer='adam', loss='mse', metrics=['mse'])

model.fit(x_train, y_train,
                 epochs=1000, batch_size=64, verbose=1,
                 validation_split=0.25,
                 callbacks=[es, cp])

# model_save, save_weights
model.save('./model/sample/boston/boston_model_save.h5')
model.save_weights('./model/sample/boston/boston_save_weights.h5')


### 4. 평가, 예측
loss, mse = model.evaluate(x_test, y_test, batch_size=64)
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

# RMSE :  4.985524460601468
# R2 :  0.7026252613187309