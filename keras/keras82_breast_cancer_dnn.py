# 20-05-31 / 일요일

# DNN 구성
# 이건 이진 분류 같음


### 1. 데이터
import numpy as np
from sklearn.datasets import load_breast_cancer
x, y = load_breast_cancer(return_X_y=True)
# print(x[0])
print(x.shape) # (569, 30)
print(y.shape) # (569,)


# 전처리
from sklearn.preprocessing import MinMaxScaler, StandardScaler
scale = MinMaxScaler()
# scale = StandardScaler()
x_s = scale.fit_transform(x)
# print(x_s[0])


# Split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x_s, y, random_state=66, shuffle=True,
    train_size=0.8)
print(x_train.shape)     # (455, 30)
print(x_test.shape)      # (114, 30)
print(y_train.shape)     # (455,)
print(y_test.shape)      # (114,)


### 2. 모델
from keras.models import Sequential
from keras.layers import Dense, Dropout
model=Sequential()
model.add(Dense(100, activation='relu', input_shape=(30,)))
model.add(Dropout(0.2))
model.add(Dense(300, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(600, activation='relu'))
model.add(Dropout(0.6))
model.add(Dense(300, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.add(Dropout(0.2))
model.summary()


### 3. 훈련
# earlystopping, modelcheckpoint(x), tensorboard(x)
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
es = EarlyStopping(monitor='val_loss', patience=50, mode='auto')

modelpath = './model/{epoch:02d}-{val_loss:4f}.hdf5'
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss',
                     save_best_only=True, mode='auto')

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
hist = model.fit(x_train, y_train,
                 epochs=55, batch_size=16, verbose=2,
                 validation_split=0.25,
                 callbacks=[es])


# matplotlib [훈련과정의 "loss와 val_loss" & "acc와 val_acc" 그래프]
import matplotlib.pyplot as plt

print(hist.history.keys())
      # dict_keys(['val_loss', 'val_acc', 'loss', 'acc'])

loss = hist.history['loss']
val_loss = hist.history['val_loss']
acc = hist.history['acc']
val_acc = hist.history['val_acc']

epochs = range(1, len(acc) + 1)     # x라벨 epochs 범위 설정
plt.figure(figsize=(10,8.5))

#1
plt.subplot(2,1,1)
plt.plot(epochs, loss, label='loss')
plt.plot(epochs, val_loss, c='red', label='val_loss')
plt.grid()
plt.title('Train : Loss & val_loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc='upper right')
#2
plt.subplot(2,1,2)
plt.plot(epochs, acc, c='blue', label='acc')
plt.plot(epochs, val_acc, c='red', label='val_acc')
plt.grid()
plt.title('Train : acc & val_acc')
plt.xlabel('Epochs')
plt.ylabel('Acc')
plt.legend(loc='lower right')

plt.subplots_adjust(hspace=0.25)    # subplot 두 그래프간 수직 간격 조정
plt.show()


### 4. 평가, 예측
loss_acc = model.evaluate(x_test, y_test, batch_size=16)
print('loss, acc : ', loss_acc)

'''
loss, acc :  [0.18103144027848253, 0.9649122953414917]
'''