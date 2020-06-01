# 20-06-01 / 월 / save 용 main 파일
# diabetes.sequential.dnn

''' < 코드 구성 목록>
 1. 'dataset'_model_save.h5
 2. 'dataset'_save_weights.h5
 3. 'dataset'_checkpoint_best.h5
'''


### 1. 데이터
import numpy as np
from sklearn.datasets import load_diabetes

x, y = load_diabetes(return_X_y=True)
print(x.shape) # (442, 10)
print(y.shape) # (442, )


### 데이터 전처리
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(x)
x_scaled = scaler.transform(x)

from sklearn.decomposition import PCA

pca = PCA(n_components=6)
pca.fit(x_scaled)
x_pca = pca.transform(x_scaled)
print(x_pca.shape)   # (442, 6)


### Train_Test_Split
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x_pca, y, random_state=66, shuffle=True,
    train_size = 0.8)
print(x_train.shape)  # (353, 6)
print(x_test.shape)   # (89, 6)
print(y_train.shape)  # (353, )
print(y_test.shape)   # (89, )


#### 2. 모델
from keras.models import Sequential
from keras.layers import Dense, Dropout

model = Sequential()

model.add(Dense(50, activation='linear', input_shape=(6,)))
model.add(Dense(100, activation='linear'))
model.add(Dropout(0.2))
model.add(Dense(200, activation='linear'))
model.add(Dropout(0.2))
model.add(Dense(300, activation='linear'))
model.add(Dropout(0.3))
model.add(Dense(200, activation='linear'))
model.add(Dropout(0.2))
model.add(Dense(100, activation='linear'))
model.add(Dropout(0.2))
model.add(Dense(50, activation='linear'))
model.add(Dense(1, activation='linear'))

model.summary()


### 3. 훈련
# earlystopping
from keras.callbacks import EarlyStopping, ModelCheckpoint

es = EarlyStopping(monitor='val_loss', patience=50, mode='auto')

# modelcheckpoint (cp_best 값 추출 할 것)
modelpath = './model/sample/diabetes/diabetes-{epoch:02d}-{val_loss:.4f}.hdf5'
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss',
                     verbose=1,
                     save_best_only=True, save_weights_only=False)

model.compile(optimizer='adam', loss='mse', metrics=['mse'])

model.fit(x_train, y_train,
                 epochs=100, batch_size=32, verbose=1,
                 validation_split=0.25,
                 callbacks=[es, cp])

# model_save, save_weights
model.save('./model/sample/diabetes/diabetes_model_save.h5')
model.save_weights('./model/sample/diabetes/diabetes_save_weights.h5')


### 4. 평가, 예측
loss_acc = model.evaluate(x_test, y_test, batch_size=32)
print(':', loss_acc)

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

'''
RMSE :  55.97069744256828
R2 :  0.5173036253026397
'''