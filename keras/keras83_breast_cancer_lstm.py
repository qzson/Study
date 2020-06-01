# 20-05-31 / 일요일

# LSTM 구성
# 이건 이진 분류 같음



### 1. 데이터
import numpy as np
from sklearn.datasets import load_breast_cancer
x, y = load_breast_cancer(return_X_y = True)
# print(x[0])
print(x.shape) # (569, 30)
print(y.shape) # (569,)

# preprocessing
from sklearn.preprocessing import MinMaxScaler, StandardScaler
scale = MinMaxScaler()
x_s = scale.fit_transform(x)

# split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x_s, y, random_state=66, shuffle=True,
    train_size=0.8)
print(x_train.shape)  # (455, 30)
print(x_test.shape)   # (114, 30)
print(y_train.shape)  # (455,)
print(y_test.shape)   # (114,)

# reshape
x_train = x_train.reshape(x_train.shape[0], 6, 5)
x_test = x_test.reshape(x_test.shape[0], 6, 5)


### 2. 모델
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

model = Sequential()
model.add(LSTM(20, activation = 'relu', input_shape = (6, 5)))
model.add(Dropout(0.4))
model.add(Dense(10, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(30, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(60, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(30, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation = 'sigmoid'))
model.summary()


### 3. 훈련
# earlystopping, modelcheckpoint, tensorboard
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
es = EarlyStopping(monitor='val_loss', patience=50, mode='auto')

modelpath = './model/{epochs:02d}-{val_loss:4f}.hdf5'
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss',
                     save_best_only=True, mode='auto')

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
hist = model.fit(x_train, y_train,
                 epochs=100, batch_size=32, verbose=2,
                 validation_split=0.25,
                 callbacks=[es])

# matplotlib
import matplotlib.pyplot as plt

print(hist.history.keys())

loss = hist.history['loss']
val_loss = hist.history['val_loss']
acc = hist.history['acc']
val_acc = hist.history['val_acc']

epochs = range(1, len(acc) + 1)
plt.figure(figsize=(10,8.5))

# 1 : loss & val_loss
plt.subplot(2,1,1)
plt.plot(epochs, loss, label='loss')
plt.plot(epochs, val_loss, c='red', label='val_loss')
plt.grid()
plt.title('Train : loss & val_loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc='upper right')
# 2 : acc & val_acc
plt.subplot(2,1,2)
plt.plot(epochs, acc, c='blue', label='acc')
plt.plot(epochs, val_acc, c='red', label='val_acc')
plt.grid()
plt.title('Train : acc & val_acc')
plt.xlabel('Epoch')
plt.ylabel('Acc')
plt.legend(loc='lower right')

plt.subplots_adjust(hspace=0.25)
plt.show()


### 4. 평가, 예측
loss_acc = model.evaluate(x_test, y_test, batch_size=32)
print('loss, acc : ', loss_acc)

'''
loss, acc :  [0.21405909609114915, 0.9561403393745422]
'''