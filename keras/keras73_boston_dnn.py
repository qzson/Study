# 20-05-29 / 1425 ~
 # data - > CSV - > pandas or numpy - > numpy -> .npy 로 저장 (중요) - > DB 에 저장
 # pandas : 자료형이 섞여있을 때
 # numpy  : 자료형이 1개일 때

'''
 .npy
 이상치와 결측치
 1. 이상치
 2. 결측치

 1. 이상치 : 1234 백만 56789101112 오백만
 2. 결측치 : 1234      56789101112 
 
 2. 결측치는 어떻게 할까? : 0 기입 or 평균값 or 사이값 or 제거 or (predict)
    (predict)           : 결측치 데이터를 x_test -> 머신돌려서 -> 결측치 자리를 predict 한다.
                          레거시한 머신러닝으로 한다. (하지만 무조건 맞다고 볼 순 없음. 그래서 여러가지를 해봐야한다.)
 
 그렇다면, 이상치는 어떻게 해결할까?
 1. 이상치

 standardscaler : 하지만 이것으로는 부족하다.

 # Robust Scaler

 - 잘 사용 X
 - 중앙값이 0, IQR이 1이 되도록 변환
 - StandardScaler에 의한 표준화보다 동일한 값을 더 넓게 분포
 - 이상치를 포함하는 데이터를 표준화하는 경우

 이전의 스케일러와는 달리, 이 스케일러의 센터링 및 스케일링 통계는 퍼센트에 근거하기 때문에,
 매우 큰 한계 아웃라이어의 수에는 영향을 받지 않습니다.
 그 결과, 변환 후의 특징량의 범위는 이전의 스케일러보다 크고,
 더욱 중요한 점은, 거의 같다 : 어느 쪽의 특징에 대해서도, 변환 후의 값의 대부분은 줌인 한 그림과 같이[-2, 3]의 범위에 있다.
 변환한 데이터에는 아웃리아 자체가 존재한다는 점에 주의해 주십시오.
 다른 외부 클리핑이 바람직한 경우에는 비선형 변환이 필요합니다
 
 # PCA
 
 차원을 축소한다. 이상치를 제거해준다.
 50개 중에 강한데이터 1개가 있다.
 그것에 맞춰서 압축해준다 (컴포넌트에 따라서)
 그래서 그 특성이 더 강해진다.
 
 '''


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
print(x_pca.shape)


###
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x_pca, y, random_state=66, shuffle=True,
    train_size = 0.8)

print(x_train.shape) #(404, 2)
print(x_test.shape)  #(102, 2)
print(y_train.shape) #(404,)
print(y_test.shape)  #(102,)


### 2. 모델
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()

model.add(Dense(100, input_shape= (2, )))
model.add(Dense(200))
model.add(Dense(300))
model.add(Dense(400))
model.add(Dense(500))
model.add(Dense(400))
model.add(Dense(300))
model.add(Dense(200))
model.add(Dense(100))
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

'''
RMSE :  6.5816520620925
R2 :  0.4817346069750055
'''