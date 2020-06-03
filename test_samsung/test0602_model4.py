# 20-06-03
# LSTM 모델 2개 앙상블 하기

# Hite 데이터를 PCA를 사용하여 (n, 1) 형태로 차원축소 해주기
# >> hite 의 거의 의미없는 '거래량' 데이터 때문, and samsung 시가 데이터 쉐이프와 맞춰준다.
# >> 앙상블[lstm1(삼성) + lstm2(하이트_PCA)] 모델 구성이 이번 시험의 핵심 의도
# p.s.) LSTM 은 파라미터간 피드백을 해주는 모델로 '거래량' 때문에 오히려 안좋은 결과를 낼 수 있다.



import numpy as np
from keras.models import Model, load_model
from keras.layers import Dense, LSTM, Dropout, Input
from keras.layers.merge import concatenate, Concatenate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA


''' 스플릿 함수 정의 '''

def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1):
        subset = seq[i:(i+size)]
        aaa.append([j for j in subset])
    # print(type(aaa))
    return np.array(aaa)
size = 6            # 6일치씩 자르겠다



''' 1. 데이터 '''

# dataset.npy 불러오기
samsung = np.load('./data/samsung_test.npy', allow_pickle='True')
hite = np.load('./data/hite_test.npy', allow_pickle='True')
 # allow_pickle 에러로 해당 인자를 추가해준다.
print(samsung.shape)                                              # (509, 1)
print(hite.shape)                                                 # (509, 5)

# 삼성 데이터 스플릿을 위한 리쉐이프 (스칼라 -> 스플릿 -> 2차원)
samsung = samsung.reshape(samsung.shape[0],)                      # (509,)

samsung = (split_x(samsung, size))
print(samsung.shape)                                              # 리쉐이프 전 (504, 6, 1) // 리쉐이프 후 # (504, 6)
                                                                  # 1개의 열을 가진 데이터 -> 6 열을 가진 데이터 (잘리며 증폭 됨)

# 삼성만 x,y를 만들어주고 하이트는 y가 필요없다. (내일의 삼성 시가 맞추기이기 때문 - 아웃풋 1개)
x_sam = samsung[:, 0:5]
y_sam = samsung[:, 5]
print(x_sam[0])                                                   # [53000 52600 52600 51700 52000]
print(y_sam[0])                                                   # 51000
print(x_sam.shape)                                                # (504, 5)
print(y_sam.shape)                                                # (504, )

# train_test_split (삼성 x, y)
x_train, x_test, y_train, y_test = train_test_split(x_sam, y_sam, train_size=0.8)
print(x_train.shape)                                              # (403, 5)
print(x_test.shape)                                               # (101, 5)
print(y_train.shape)                                              # (403,)
print(y_test.shape)                                               # (101,)

# 스케일링 (삼성 x, y)
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(x_train[0, :])               # [-0.02863956 -0.09552053 -0.21724087 -0.29874723 -0.5847868 ]
    # 대단히 주의할 점 : 데이터를 나누고 scale 하는 것이 좋으며, scale.fit은 train이 기준이 되어야 한다.
    #  >> 나머지 (test, pred 들은 scale.fit 된, train이 기준이 되어야 함으로 transform만 진행한다.)
    #  >> 만약, scale 후 데이터를 나눴다면, 결과값이 좋지는 않겠지만, 추후 pred 기입 시, pred는 따로 scale.transfrom 해야한다.
    #    >> 그렇지 않으면, 결과 예측 값이 부정확할 것이다.



# 스케일링 PCA (하이트 전체 데이터) // 최상단도 설명 기재
 # Hite 데이터를 PCA를 사용하여 (n, 1) 형태로 차원축소 해주기
 # >> hite 의 거의 의미없는 '거래량' 데이터 때문, and samsung 시가 데이터 쉐이프와 맞춰준다.
 # >> 앙상블[lstm1(삼성) + lstm2(하이트_PCA)] 모델 구성이 이번 시험의 핵심 의도
 # p.s.) LSTM 은 파라미터간 피드백을 해주는 모델로 '거래량' 때문에 오히려 안좋은 결과를 낼 수 있다.
scaler = StandardScaler()
scaler.fit(hite)
hite_scaled = scaler.transform(hite)
# print(hite_scaled)

pca = PCA(n_components=1)
pca.fit(hite_scaled)
hite_pca = pca.transform(hite_scaled)
print(hite_pca.shape)                                             # (509, 1)

# 스플릿 함수 적용
x_hit = (split_x(hite_pca, size))
print(x_hit.shape)                                                # (504, 6, 1)

# train_test_split (하이트 x)
x2_train, x2_test = train_test_split(x_hit, train_size=0.8)
print(x2_train.shape)                                             # (403, 6, 1)
print(x2_test.shape)                                              # (101, 6, 1)


# LSTM 구성 위해서 리쉐이프
x_train = x_train.reshape(x_train.shape[0],5,1)                   # (403, 5, 1)
x_test = x_test.reshape(x_test.shape[0],5,1)                      # (101, 5, 1)



''' 2. 모델 '''

input1 = Input(shape=(5,1))
x1 = LSTM(100)(input1)
x1 = Dense(1000)(x1)
x1 = Dense(100)(x1)

input2 = Input(shape=(6,1))
x2 = LSTM(100)(input2)
x2 = Dense(1000)(x2)
x2 = Dense(100)(x2)

merge = concatenate([x1, x2])

output = Dense(1)(merge)

model = Model(inputs=[input1, input2], outputs=output)

model.summary()



''' 3. 실행, 훈련 '''

model.compile(optimizer='adam', loss='mse')

model.fit([x_train, x2_train], y_train, epochs=200, batch_size=32, verbose=1, validation_split=0.2)



''' 4. 평가, 예측 '''

mse = model.evaluate([x_test, x2_test], y_test, batch_size=32)
print('mse', mse)

y_pred = model.predict([x_test, x2_test])
print(y_pred)
