# 20-06-08_20
# mon / 16:00 ~

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Dropout, Input
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from keras.utils import np_utils
from sklearn.pipeline import Pipeline, make_pipeline


''' 1. 데이터 '''

train = pd.read_csv('./data/dacon/comp1/train.csv', header=0, index_col=0)
test = pd.read_csv('./data/dacon/comp1/test.csv', header=0, index_col=0)
submission = pd.read_csv('./data/dacon/comp1/sample_submission.csv', header=0, index_col=0)

print('train.shape :', train.shape)             # (10000, 75) : x_train, test
print('test.shape :', test.shape)               # (10000, 71) : x_predict
print('submission.shape :', submission.shape)   # (10000,  4) : y_predict

'''
train.csv >> x_train,test / y_train,test 로 나누고
test.csv  >> x_pred
subm.csv  >> y_pred                      로 나눈다

train.csv 데이터를 보니, 컬럼 마지막 부분에 subm의 4가지 y값이 이쁘게 들어 있었으므로
그리고 아웃풋이 4개다
회귀
'''

# 결측치가 어떤놈인지 함 보자 (컬럼당 얼마나 모여있는지)
# print(train.isnull().sum())      # 일단 있는 것을 확인
train = train.interpolate()      # 보간법 // 선형보간 (검색하면 나온다)
# print(train.isnull().sum())    # 빈자리를 컬럼별로 선을 그려서 빈자리에 알아서 채워준다 (값들 전체에 대해 선을 그리고 중간마다 듬성 듬성 다 넣어준다)
test = test.interpolate()
# print(train.head())
# print(train.shape)
# print(test.shape)

train_npy = train.values
test_npy = test.values
# print(type(train_npy)) # npy
# print(type(test_npy)) # npy

x = train_npy[:, :-4]
y = train_npy[:, -4:]
# print(x[0])
# print(y[0])
# print(x.shape) # 10000, 71
# print(y.shape) # 10000, 4

# 스플릿
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 66, shuffle = True)
print(x_train.shape) # (8000, 71)
print(x_test.shape)  # (2000, 71)
print(y_train.shape) # (8000, 4)
print(y_test.shape)  # (2000, 4)

# x전처리
sc = MinMaxScaler()
sc.fit(x_train)
x_train = sc.transform(x_train)
x_test = sc.transform(x_train)


''' 2. 모델 '''
def build_model(drop=0.5, optimizer='adam'):
    inputs = Input(shape=(71,), name='input')
    x = Dense(1024, activation='relu', name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(512, activation='relu', name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(256, activation='relu', name='hidden3')(x)
    x = Dropout(drop)(x)
    output1 = Dense(64, activation='relu', name='outputs')(x)
    output2 = Dense(64, activation='relu', name='outputs')(x)
    output3 = Dense(64, activation='relu', name='outputs')(x)
    model = Model(inputs=inputs, outputs=[output1, output2, output3])
    model.compile(optimizer=optimizer, metrics=['mae'],
                  loss='mae')
    return model

def create_hyperparameters():
    batches = [10, 20, 30, 40, 50]
    optimizers = ['adam']
    dropout = np.linspace(0.1, 0.5, 5)
    return{"batch_size" : batches, "optimizer" : optimizers,
           "drop" : dropout}

from keras.wrappers.scikit_learn import KerasRegressor
model = KerasRegressor(build_fn=build_model, verbose=1)
hyperparameters = create_hyperparameters()

from sklearn.model_selection import RandomizedSearchCV
search = RandomizedSearchCV(model, hyperparameters, cv=3)
search.fit(x_train, y_train)
print('최적의 매개변수 :', search.best_params_)

mae = search.score(x_test, y_test)
print('최종 스코어 :', mae)







# ''' 3. 훈련 '''





# ''' 4. 평가, 예측 '''





# # submit 열어보면 왼쪽에 인덱스 위쪽에 헤더
# # y_pred.to_csv(경로) : pred할 submit 파일을 만든다

