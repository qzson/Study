# 20-07-02_27
# optimizer, learning_rate 추가했다.
# dnn 구성


from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Input, Dropout, Conv2D, Flatten
from keras.layers import Dense, MaxPooling2D, LSTM
from keras.optimizers import Adam, RMSprop, SGD, adadelta, adagrad, nadam
import numpy as np


''' 1. 데이터 '''

# data 불러오기
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape)    # 60000, 28, 28
print(x_test.shape)     # 10000, 28, 28

# data 전처리, 리쉐이프
x_train = x_train.reshape(x_train.shape[0], 28 * 28).astype('float')/255
x_test = x_test.reshape(x_test.shape[0], 28 * 28).astype('float')/255
print(x_train.shape)    # (60000, 784)
print(x_test.shape)     # (10000, 784)

# y 원핫인코딩 (차원 확인할 것)
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train.shape)    # 60000, 10
print(y_test.shape)     # 10000, 10


''' 2. 모델 '''
def build_model(drop=0.5, optimizer=Adam, lr=0.01):
    inputs = Input(shape=(28 * 28,), name='input')
    x = Dense(64, activation='relu', name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(32, activation='relu', name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(16, activation='relu', name='hidden3')(x)
    x = Dropout(drop)(x)
    outputs = Dense(10, activation='softmax', name='outputs')(x)
    model = Model(inputs=inputs, outputs=outputs)

    opt = optimizer(lr=lr)

    model.compile(optimizer=opt, metrics=['acc'],
                  loss='categorical_crossentropy')
    return model


def create_hyperparameters():
    batches = [128, 256]
    optimizers = [Adam, RMSprop, SGD, adadelta, adagrad, nadam]
    learning_rate = [0.1, 0.01, 0.001]
    dropout = np.linspace(0.1, 0.5, 5)
    return{"batch_size" : batches, "optimizer" : optimizers,
           "drop" : dropout, "lr" : learning_rate}

from keras.wrappers.scikit_learn import KerasClassifier
model = KerasClassifier(build_fn=build_model, verbose=2)
hyperparameters = create_hyperparameters()

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
search = RandomizedSearchCV(model, hyperparameters, cv=3)
search.fit(x_train, y_train)
print('최적의 매개변수 :', search.best_params_)

acc = search.score(x_test, y_test)
print('최종 스코어 :', acc)


# pinrt 값 (node : 128, 64, 32)
# 최적의 매개변수 : {'optimizer': 'adadelta', 'drop': 0.1, 'batch_size': 10}
# 최종 스코어 : 0.9240999817848206
