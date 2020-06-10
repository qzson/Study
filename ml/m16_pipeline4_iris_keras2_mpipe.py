# 20-06-10_22 수요일 // 1020~

# iris를 keras pipeline으로 구성
# 당연히 RandomizedSearchCV 구성
# keras98 참고할 것
# make_pipeline으로 구성

import pandas as pds
import numpy as np

from sklearn.datasets import load_iris
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Input, Dropout, Conv2D, Flatten
from keras.layers import Dense, MaxPooling2D

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.layers import LeakyReLU
leaky = LeakyReLU(alpha = 0.2)


''' 1. 데이터 '''

iris = load_iris()
x = iris.data
y = iris.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 66, shuffle = True)
print(x_train.shape)    # (120, 4)
print(x_test.shape)     # (30, 4)

# y 원핫인코딩 (차원 확인할 것)
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train.shape)    # (120, 3)
print(y_test.shape)     # (30, 3)


''' 2. 모델 '''

def build_model(drop=0.5, optimizer='adam', act='relu'):
    inputs = Input(shape=(4,), name='input')
    x = Dense(512, activation=act, name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(256, activation=act, name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(128, activation=act, name='hidden3')(x)
    x = Dropout(drop)(x)
    outputs = Dense(3, activation='softmax', name='outputs')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, metrics=['acc'],
                  loss='categorical_crossentropy')
    return model

def create_hyperparameters():
    batches = [128, 256, 512]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    # dropout = [0.1, 0.2, 0.3, 0.4]
    dropout = np.linspace(0.1, 0.3, 3).tolist()
    epochs = [30, 50, 70, 100]
    activation = ['relu', 'elu', leaky]
    return{"kerasclassifier__batch_size" : batches, "kerasclassifier__optimizer" : optimizers,
           "kerasclassifier__drop" : dropout, "kerasclassifier__epochs" : epochs, "kerasclassifier__act" : activation
           }

from keras.wrappers.scikit_learn import KerasClassifier
model = KerasClassifier(build_fn=build_model, verbose=1)
hyperparameters = create_hyperparameters()

pipe = make_pipeline(MinMaxScaler(), model)
#   >> make_pipeline은 위 parameters의 return 요소 정의에 'model__'을 쓰면 안되고 'kerasclassifier__' 를 넣어줘야 한다

search = RandomizedSearchCV(pipe, hyperparameters, cv=3)
search.fit(x_train, y_train)
print('최적의 매개변수 :', search.best_params_)

acc = search.score(x_test, y_test)
print('최종 스코어 :', acc)


# 최적의 매개변수 : {'KR__optimizer': 'adam', 'KR__epochs': 70, 'KR__drop': 0.3, 'KR__batch_size': 256}
# 최종 스코어 : 0.9333333373069763

# dropout = [0.1, 0.3, 0.5]
# 최적의 매개변수 : {'KR__optimizer': 'rmsprop', 'KR__epochs': 70, 'KR__drop': 0.2, 'KR__batch_size': 128, 'KR__act': 'relu'}
# 최종 스코어 : 0.8666666746139526

## dropout = np.linspace(0.1, 0.3, 3).tolist()
# 최적의 매개변수 : {'KR__optimizer': 'adadelta', 'KR__epochs': 70, 'KR__drop': 0.2, 'KR__batch_size': 128, 'KR__act': 'relu'}
# 최종 스코어 : 0.6333333253860474

## make_pipeline / kerasclassifier__
# 최적의 매개변수
#  : {'kerasclassifier__optimizer': 'adadelta', 'kerasclassifier__epochs': 100,
# 'kerasclassifier__drop': 0.3, 'kerasclassifier__batch_size': 128, 'kerasclassifier__act': 'elu'}
# 최종 스코어 : 0.8666666746139526
