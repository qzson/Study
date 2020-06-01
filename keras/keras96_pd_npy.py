# 95번을 불러와서 모델을 완성하세요.


### 1. 데이터

# 데이터 load
import numpy as np

aaa = np.load('./data/iris_pd.npy')
print(aaa.shape) # (150, 5)

# 데이터 x, y 할당 및 슬라이싱
x = aaa[:, :4]
# print(x)
print(x.shape) # (150, 4)

y = aaa[:, 4]
# 혹은, y = aaa[:, 4:]
# print(y)
# y 값은 순서대로 0 ~ 2 가 나오니 split 할 때 shuffle 을 넣어줘야 결과치가 제대로 나온다.
print(y.shape) # (150, )


# 전처리 1. PCA
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(x)
x_scaled = scaler.transform(x)
# print(x_scaled)

from sklearn.decomposition import PCA

pca = PCA(n_components=3)
pca.fit(x_scaled)
x_pca = pca.transform(x_scaled)
# print(x_pca)
print(x_pca.shape)   #(150, 3)


# 전처리 2. Train_Test_Split
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x_pca, y, random_state=66, shuffle=True,
    train_size = 0.8)
print(x_train.shape) #(120, 3)
print(x_test.shape)  #(30, 3)
print(y_train.shape) #(120,)
print(y_test.shape)  #(30,)


# 전처리 3. OneHotEncoding (이것을 Train_Test_split 전에 하면 y만 하면됨)
from keras.utils import np_utils

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train.shape) #(120, 3)
print(y_test.shape)  #(30, 3)


### 2. 모델
from keras.models import Sequential
from keras.layers import Dense, Dropout

model = Sequential()

model.add(Dense(50, activation='relu', input_shape = (3, )))
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
# modelpath = './model/sample/iris/iris-{epoch:02d}-{val_loss:.4f}.hdf5'
# cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss',
#                      verbose=1,
#                      save_best_only=True, save_weights_only=False)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])

model.fit(x_train, y_train,
                 epochs=100, batch_size=8, verbose=1,
                 validation_split=0.25,
                 callbacks=[es])


### 4. 평가, 예측
loss_acc = model.evaluate(x_test, y_test, batch_size=8)
print('loss_acc : ', loss_acc)
# loss_acc :  [0.49479877288298063, 0.9333333373069763]