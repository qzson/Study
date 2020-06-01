# 20-06-01 / 월 / save 용 main 파일
# fashion_mnist.sequential.CNN

''' < 코드 구성 목록>
 1. 'dataset'_model_save.h5
 2. 'dataset'_save_weights.h5
 3. 'dataset'_checkpoint_best.h5
'''

### 1. 데이터

# 데이터셋 불러오기
import numpy as np
from keras.datasets import fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
print(x_train.shape)                # (60000, 28, 28)
print(x_test.shape)                 # (10000, 28, 28)
print(y_train.shape)                # (60000,)
print(y_test.shape)                 # (10000,)

# 데이터 전처리 1. OneHotEncoding
from keras.utils import np_utils

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train.shape)                 # (60000, 10)

# 데이터 전처리 2. 정규화(MinMaxScaler)
x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255


### 2. 모델
from keras.models import Sequential
from keras.layers import Dense, Conv2D
from keras.layers import Flatten, MaxPooling2D, Dropout

model = Sequential()
model.add(Conv2D(10, (2,2), activation='relu', padding='same', input_shape=(28,28,1)))
model.add(Conv2D(40, (2,2), activation='relu', padding='same'))
model.add(Conv2D(70, (2,2), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(50, (2,2), activation='relu', padding='same'))
model.add(Conv2D(40, (2,2), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(30, (2,2), activation='relu', padding='same'))
model.add(Conv2D(20, (2,2), activation='relu', padding='same'))
model.add(Conv2D(10, (2,2), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(10, (2,2), activation='relu', padding='same'))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

model.summary()


### 3. 훈련
# earlystopping
from keras.callbacks import EarlyStopping, ModelCheckpoint

es = EarlyStopping(monitor='val_loss', patience=20, mode='auto')

# modelcheckpoint (cp_best 값 추출 할 것)
modelpath = './model/sample/fashion/fashion-{epoch:02d}-{val_loss:.4f}.hdf5'
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss',
                     verbose=1,
                     save_best_only=True, save_weights_only=False)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])

model.fit(x_train, y_train,
                 epochs=10, batch_size=64, verbose=1,
                 validation_split=0.25,
                 callbacks=[es, cp])

# model_save, save_weights
model.save('./model/sample/fashion/fashion_model_save.h5')
model.save_weights('./model/sample/fashion/fashion_save_weights.h5')


# 4. 평가, 예측
loss_acc = model.evaluate(x_test, y_test, batch_size=64)
print('loss_acc : ', loss_acc)
# loss_acc :  [0.28134672648906706, 0.9002000093460083]