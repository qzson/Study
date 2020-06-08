# 20-06-08_20
# 월요일 // 11:30 ~

# keras97 을 RandomizedSearchCV로 변경
# score 넣어서 score 까지


from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Input, Dropout, Conv2D, Flatten
from keras.layers import Dense, MaxPooling2D
import numpy as np


''' 1. 데이터 '''

# data 불러오기
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape)    # 60000, 28, 28
print(x_test.shape)     # 10000, 28, 28

# data 전처리, 리쉐이프
x_train = x_train.reshape(x_train.shape[0], 28*28).astype('float')/255
x_test = x_test.reshape(x_test.shape[0], 28*28).astype('float')/255
print(x_train.shape)    # (60000, 784)
print(x_test.shape)     # (10000, 784)

# y 원핫인코딩 (차원 확인할 것)
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train.shape)    # 60000, 10
print(y_test.shape)     # 10000, 10


''' 2. 모델 '''

def build_model(drop=0.5, optimizer='adam'):
    inputs = Input(shape=(28*28,), name='input')
    x = Dense(512, activation='relu', name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(256, activation='relu', name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(128, activation='relu', name='hidden3')(x)
    x = Dropout(drop)(x)
    outputs = Dense(10, activation='softmax', name='outputs')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, metrics=['acc'],
                  loss='categorical_crossentropy')
    return model

def create_hyperparameters():
    batches = [10, 20, 30, 40, 50]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = np.linspace(0.1, 0.5, 5)
    return{"batch_size" : batches, "optimizer" : optimizers,
           "drop" : dropout}

from keras.wrappers.scikit_learn import KerasClassifier
model = KerasClassifier(build_fn=build_model, verbose=1)
hyperparameters = create_hyperparameters()

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
search = RandomizedSearchCV(model, hyperparameters, cv=3)
search.fit(x_train, y_train)
print('최적의 매개변수 :', search.best_params_)

acc = search.score(x_test, y_test)
print('최종 스코어 :', acc)

# model.best_estimator_ : 위에 것과 비교해서 큰 의미는 없는데, 다른 것은 아니다. 근데 값이 다른가?
# 시스템 상으로 Grid-CV, Random-CV를 사용해서 최적의 값을 추출하는 것이지만,
# 그것을 토대로 튜닝을 해도 막상 최적의 값이 아닐 '수'도 있다
# 넣을 수 있는 것 : epoch, node의 개수, activation, ...경사하강법에 LR? 도 쓸 수 있다
#   >> hyperparameter에 명시해주고 필요한 부분 사용하면 된다

# pinrt 값
# 1. Default RandomizedSearchCV result : {'optimizer': 'adadelta', 'drop': 0.1, 'batch_size': 40}
# 2. Score 추가                        : {'optimizer': 'rmsprop', 'drop': 0.1, 'batch_size': 30} // 0.9674999713897705