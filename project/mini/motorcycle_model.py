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
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
history = model.fit(x_train, y_train, epochs=50, batch_size=8, verbose=1, validation_split=0.2)


''' 4. 평가, 예측 '''
loss, acc = model.evaluate(x_test, y_test, batch_size=8)
print("loss :", loss)
print("acc :", acc)


y_vloss = history.history['val_loss']
y_loss = history.history['loss']

x_len = np.arange(len(y_loss))

plt.plot(x_len, y_vloss, marker='.', c='red', label='val_set_loss')
plt.plot(x_len, y_loss, marker='.', c='blue', label='train_set_oss')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('loss')
plt.grid()
plt.show()