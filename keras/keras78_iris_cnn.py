# 20-05-30 / 토요일

# 아이리스 CNN 모델 구성
# 요노마는 꽃 3개를 분류하는 데이터 인가 보다
# 그러므로 다중 분류 들어갈 것임


### 1. 데이터
import numpy as np
from sklearn.datasets import load_iris

x, y = load_iris(return_X_y=True)

print(x.shape) # (150, 4)
print(y.shape) # (150,)


### PCA
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


### Train_Test_Split
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x_pca, y, random_state=66, shuffle=True,
    train_size = 0.8)

print(x_train.shape) #(120, 2)
print(x_test.shape)  #(30, 2)
print(y_train.shape) #(120,)
print(y_test.shape)  #(30,)


### x.Reshape
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1, 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1, 1)


### OneHotEncoding
from keras.utils import np_utils

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train.shape) #(120, 3)
print(y_test.shape)  #(30, 3)


### 2. 모델
from keras.models import Sequential
from keras.layers import Dense, Conv2D, LSTM, Dropout
from keras.layers import Dropout, MaxPooling2D, Flatten

model = Sequential()

model.add(Conv2D(10, (1,1), activation='relu', padding='same', input_shape=(2,1,1)))
model.add(Conv2D(40, (1,1), activation='relu', padding='same'))
model.add(Conv2D(70, (1,1), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(50, (1,1), activation='relu', padding='same'))
model.add(Conv2D(40, (1,1), activation='relu', padding='same'))
# model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(30, (1,1), activation='relu', padding='same'))
model.add(Conv2D(20, (1,1), activation='relu', padding='same'))
model.add(Conv2D(10, (1,1), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(10, (1,1), activation='relu', padding='same'))
model.add(Flatten())
model.add(Dense(3, activation='softmax'))

model.summary()


##### EarlyStopping & Modelcheckpoint
from keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='loss', patience=50)

modelpath = './model/{epoch:02d}-{val_loss:.4f}.hdf5'
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss',
                     save_best_only=True, mode='auto')


##### 3. 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
hist = model.fit(x_train, y_train,
          epochs=100, batch_size=32, verbose=2,
          validation_split=0.4,
          callbacks=[es, cp])


##### 4. 평가, 예측
loss_acc = model.evaluate(x_test, y_test, batch_size=32)

loss = hist.history['loss']
val_loss = hist.history['val_loss']
acc = hist.history['acc']
val_acc = hist.history['val_acc']

print('acc 는 ', acc)
print('val_acc 는 ', val_acc)

# evaluate 종속 결과
print('loss, acc 는 ', loss_acc)


##### plt 시각화
import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))

# 1
plt.subplot(2,1,1)
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid()
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')

# 2
plt.subplot(2,1,2)
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.grid()
plt.title('acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['acc', 'val_acc'])

plt.show()

'''
loss, acc 는  [0.4518324136734009, 0.800000011920929]
'''