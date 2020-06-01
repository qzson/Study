# 20-06-01 / 월 / save 용 main 파일
# breast_cancer.sequential.dnn

''' < 코드 구성 목록>
 1. 'dataset'_model_save.h5
 2. 'dataset'_save_weights.h5
 3. 'dataset'_checkpoint_best.h5
'''


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
model.summary()


### 3. 훈련
# earlystopping
from keras.callbacks import EarlyStopping, ModelCheckpoint

es = EarlyStopping(monitor='val_loss', patience=20, mode='auto')

# modelcheckpoint (cp_best 값 추출 할 것)
modelpath = './model/sample/breast_cancer/cancer-{epoch:02d}-{val_loss:.4f}.hdf5'
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss',
                     verbose=1,
                     save_best_only=True, save_weights_only=False)

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

model.fit(x_train, y_train,
                 epochs=55, batch_size=16, verbose=1,
                 validation_split=0.25,
                 callbacks=[es, cp])

# model_save, save_weights
model.save('./model/sample/breast_cancer/cancer_model_save.h5')
model.save_weights('./model/sample/breast_cancer/cancer_save_weights.h5')


### 4. 평가, 예측
loss_acc = model.evaluate(x_test, y_test, batch_size=16)
print('loss_acc : ', loss_acc)
# loss_acc :  [0.49489907821075657, 0.9561403393745422]