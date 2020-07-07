# 20-07-06_29
# self 실습 모델

from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model


### 1. data
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=5000)

max_len = 500
x_train = pad_sequences(x_train, maxlen=max_len)
x_test = pad_sequences(x_test, maxlen=max_len)

# print(x_train)
# print(x_train.shape)    # (25000, 500)
# print(x_test.shape)     # (25000, 500)
# print(y_train.shape)      # (25000)
# print(y_test.shape)       # (25000)


### 2. model
model = Sequential()

model.add(Embedding(5000, 120))
model.add(LSTM(120))
model.add(Dense(1, activation='sigmoid'))

model.summary()


### 3. compile, fit
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
mc = ModelCheckpoint('./model/k126_{epoch:02d}_{val_acc:.3f}.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=64, callbacks=[es, mc])

# loss: 0.1767 - acc: 0.9335 - val_loss: 
# 0.3567 - val_acc: 0.8656
