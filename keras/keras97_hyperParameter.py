# 20-06-08_20
# 월요일 // 10:00 ~

# gridsearch, randomsearch 사용
# Dense 모델

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
# x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float')/255
# x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float')/255
    # >> 0~255까지 들어가 있는 것을 255으로 나누면 minmax와 같다
    # >> CNN 용
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

# GridSearch를 사용할 것이다 그것을 쓰기 위해서 함수제작
# model = GridSearchCV(RandomForestClassifier(), parameters, cv=kfold)
#   >> 모델이 들어가는 진짜 함수를 만들 것이다

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
#   >> 함수형 모델과 동일하고 그걸 단지 함수로 감싼 다음에 함수에 들어갈 매개변수 drop, optimizer를 넣어놓고 return 시킨 것이다
#   >> for문 돌리면 2~3개도 쓸 수 있다. (def 윗줄에 for문 들어가면)
#   >> fit은 어디에 ? : grids or randoms에서

def create_hyperparameters():
    batches = [10, 20, 30, 40, 50]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = np.linspace(0.1, 0.5, 5)
    return{"batch_size" : batches, "optimizer" : optimizers,
           "drop" : dropout}
#   >> grids 에 들어갈 요소들이 준비 되었다
#   >> keras에 그냥 사용해서는 안되고 keras의 sklearn의 wrapers class를 땡겨온다
#   >> hyperparameters의 return은 딕셔너리 형태

from keras.wrappers.scikit_learn import KerasClassifier
# keras건 sklearn 이건 분류와 회귀가 있다는 점 잊지말자
model = KerasClassifier(build_fn=build_model, verbose=1)
hyperparameters = create_hyperparameters()

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
search = GridSearchCV(model, hyperparameters, cv=3)
# estimator) sklearn에 쓸 수 있게 wraping을 한 것이다 (위 build_model def와 model 명시 확인)
search.fit(x_train, y_train)

print(search.best_params_)
# batch_size에서 5번, optimizer 3번, dropout 5번, cv 3번 해서 = 총 255번 연산 (그래서 오래걸린다)