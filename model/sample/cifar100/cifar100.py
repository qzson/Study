# 20-06-01 / 월 / save 용 main 파일
# cifar100.hamsu.CNN

''' < 코드 구성 목록>
 1. 'dataset'_model_save.h5
 2. 'dataset'_save_weights.h5
 3. 'dataset'_checkpoint_best.h5
'''

### 1. 데이터
import numpy as np
from keras.datasets import cifar100

(x_train, y_train), (x_test, y_test) = cifar100.load_data()
print(x_train.shape)        # (50000, 32, 32, 3)
print(x_test.shape)         # (10000, 32, 32, 3)
print(y_train.shape)        # (50000, 1)
print(y_test.shape)         # (10000, 1)

# 전처리 1. OneHotEncoding
from keras.utils import np_utils

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train.shape)                 # (50000, 100)

# 전처리 2. 리쉐이프 & 정규화
x_train = x_train.reshape(50000, 32, 32, 3).astype('float32')/255
x_test = x_test.reshape(10000, 32, 32, 3).astype('float32')/255


### 2. 모델
from keras.models import Model
from keras.layers import Dense, Conv2D, Input
from keras.layers import Flatten, MaxPooling2D, Dropout

input1 = Input(shape=(32,32,3))

Conv1 = Conv2D(10, (3,3), activation='relu', padding='same', name='Conv1')(input1)
dp1 = Dropout(0.2)(Conv1)
Conv1 = Conv2D(40, (3,3), activation='relu', padding='same', name='Conv2')(dp1)
dp1 = Dropout(0.2)(Conv1)
Conv1 = Conv2D(70, (3,3), activation='relu', padding='same', name='Conv3')(dp1)
mx1 = MaxPooling2D(2,2)(Conv1)
Conv1 = Conv2D(50, (3,3), activation='relu', padding='same', name='Conv4')(mx1)
dp1 = Dropout(0.2)(Conv1)
Conv1 = Conv2D(40, (3,3), activation='relu', padding='same', name='Conv5')(dp1)
mx1 = MaxPooling2D(2,2)(Conv1)
Conv1 = Conv2D(30, (3,3), activation='relu', padding='same', name='Conv6')(mx1)
dp1 = Dropout(0.2)(Conv1)
Conv1 = Conv2D(20, (3,3), activation='relu', padding='same', name='Conv7')(dp1)
dp1 = Dropout(0.2)(Conv1)
Conv1 = Conv2D(10, (3,3), activation='relu', padding='same', name='Conv8')(dp1)
Conv1 = Conv2D(10, (3,3), activation='relu', padding='same', name='Conv9')(Conv1)
ft1 = Flatten()(Conv1)
output1 = Dense(100, activation='softmax', name='output1')(ft1)

model = Model(inputs=input1, outputs=output1)

model.summary()


### 3. 훈련
# earlystopping
from keras.callbacks import EarlyStopping, ModelCheckpoint

es = EarlyStopping(monitor='val_loss', patience=20, mode='auto')

# modelcheckpoint (cp_best 값 추출 할 것)
modelpath = './model/sample/cifar100/cifar100-{epoch:02d}-{val_loss:.4f}.hdf5'
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss',
                     verbose=1,
                     save_best_only=True, save_weights_only=False)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])

model.fit(x_train, y_train,
                 epochs=30, batch_size=256, verbose=1,
                 validation_split=0.25,
                 callbacks=[es, cp])

# model_save, save_weights
model.save('./model/sample/cifar100/cifar100_model_save.h5')
model.save_weights('./model/sample/cifar100/cifar100_save_weights.h5')


### 4. 평가, 예측
loss_acc = model.evaluate(x_test, y_test, batch_size=256)
print('loss_acc : ', loss_acc)
# loss_acc :  [2.8054678115844727, 0.31200000643730164]