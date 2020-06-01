# 20-06-01 / 월 / save 용 main 파일
# iris.sequential.dnn

''' < 코드 구성 목록>
 1. 'dataset'_model_save.h5
 2. 'dataset'_save_weights.h5
 3. 'dataset'_checkpoint_best.h5
'''

### 1. 데이터
# 데이터 불러오기
import numpy as np
from sklearn.datasets import load_iris

x, y = load_iris(return_X_y=True)
print(x.shape) # (150, 4)
print(y.shape) # (150,)

# 전처리 1. PCA
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(x)
x_scaled = scaler.transform(x)
# print(x_scaled)

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca.fit(x_scaled)
x_pca = pca.transform(x_scaled)
# print(x_pca)
print(x_pca.shape)   #(150, 2)

# 전처리 2. Train_Test_Split
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x_pca, y, random_state=66, shuffle=True,
    train_size = 0.8)
print(x_train.shape) #(120, 2)
print(x_test.shape)  #(30, 2)
print(y_train.shape) #(120,)
print(y_test.shape)  #(30,)

# 전처리 3. OneHotEncoding
from keras.utils import np_utils

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train.shape) #(120, 3)
print(y_test.shape)  #(30, 3)


### 2. 모델
from keras.models import Sequential
from keras.layers import Dense, Dropout

model = Sequential()

model.add(Dense(50, activation='relu', input_shape = (2, )))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(200, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(300, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(3, activation='softmax'))

model.summary()


### 3. 훈련
# earlystopping
from keras.callbacks import EarlyStopping, ModelCheckpoint

es = EarlyStopping(monitor='val_loss', patience=20, mode='auto')

# modelcheckpoint (cp_best 값 추출 할 것)
modelpath = './model/sample/iris/iris-{epoch:02d}-{val_loss:.4f}.hdf5'
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss',
                     verbose=1,
                     save_best_only=True, save_weights_only=False)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])

model.fit(x_train, y_train,
                 epochs=400, batch_size=32, verbose=1,
                 validation_split=0.25,
                 callbacks=[es, cp])

# model_save, save_weights
model.save('./model/sample/iris/iris_model_save.h5')
model.save_weights('./model/sample/iris/iris_save_weights.h5')


### 4. 평가, 예측
loss_acc = model.evaluate(x_test, y_test, batch_size=32)
print('loss_acc : ', loss_acc)
# loss_acc :  [0.2648666203022003, 0.8666666746139526]