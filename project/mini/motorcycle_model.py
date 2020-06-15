# CNN 모델 구성 및 checkpoint 저장

import os, glob, numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt


''' 1. 데이터 '''
x_train, x_test, y_train, y_test = np.load('./project/mini/data/multi_image_data.npy')
print(x_train.shape)    # (160, 100, 100, 3)
print(x_test.shape)     # (40, 100, 100, 3)
print(y_train.shape)    # (160, 4)
print(y_test.shape)     # (40, 4)

categories = ['scooter', 'supersports', 'multipurpose', 'cruiser']
nb_classes = len(categories)

x_train = x_train.astype(float)/255
x_test = x_test.astype(float)/255


''' 2. 모델 '''
model = Sequential()

model.add(Conv2D(32, (3,3), activation='relu', padding='same', input_shape=x_train.shape[1:]))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(nb_classes, activation='softmax'))

model.summary()


''' 3. 훈련 '''
## earlystopping & modelcheckpoint
es = EarlyStopping(monitor='val_loss', patience=7, mode='auto')

modelpath = './project/mini/checkpoint/cp-{epoch:02d}-{val_loss:.4f}.hdf5'
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
hist = model.fit(x_train, y_train, epochs=40, batch_size=8, verbose=1, validation_split=0.2, callbacks=[es])


''' 4. 평가 '''
loss, acc = model.evaluate(x_test, y_test, batch_size=8)
print("loss :", loss)
print("acc :", acc)

## pyplot 시각화
y_vloss = hist.history['val_loss']
y_loss = hist.history['loss']
y_vacc = hist.history['val_acc']
y_acc = hist.history['acc']

x_len1 = np.arange(len(y_loss))
x_len2 = np.arange(len(y_acc))
plt.figure(figsize=(6,6))

## 1 Loss 그래프
plt.subplot(2,1,1)
plt.plot(x_len1, y_vloss, marker='.', c='red', label='val_loss')
plt.plot(x_len1, y_loss, marker='.', c='blue', label='loss')
plt.legend()
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.grid()

## 2 Acc 그래프
plt.subplot(2,1,2)
plt.plot(x_len2, y_vacc, marker='.', c='orange', label='val_acc')
plt.plot(x_len2, y_acc, marker='.', c='purple', label='acc')
plt.legend()
plt.title('Acc')
plt.xlabel('epochs')
plt.ylabel('acc')
plt.grid()

plt.subplots_adjust(hspace=0.4)
plt.show()
