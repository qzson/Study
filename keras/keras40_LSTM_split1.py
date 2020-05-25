# 200525 1000~
# LSTM 모델을 완성하시오.

''' 튜닝 값
    loss: 4.2007144656963646e-05
 [[4.996726 ]
 [5.9957223]
 [7.00133  ]
 [8.007996 ]
 [9.007314 ]
 [9.98981  ]]
'''

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM

# 1. 데이터
a = np.array(range(1, 11))
size = 5

""" split 정의 함수로 x,y 쪼개기
def split_x(seq, size):
    xxx=[]
    for i in range(len(seq) - size + 1):       # len = length  : 길이  i in range(6)  : [0, 1, 2, 3, 4, 5]
        subset = seq[i : (i + size -1)]    
                                               # i =0,  subset = a[ 0 : 5 ] = [ 1, 2, 3, 4, 5]
        xxx.append([item for item in subset])  # aaa = [[1, 2, 3, 4, 5]]
        #aaa.append([subset])          
    return np.array(xxx)
def split_y(seq, size):
    yyy = []
    for i in range(len(seq) - size + 1):      
        y = seq[(i + size-1)]
        print(y)             
        yyy.append(y) 
    return np.array(yyy)
 x = split_x(a, size)                       
 print(x.shape)                                # (6, 4)   # time_steps = 4
 y = split_y(a, size)                       
 print(y.shape)                                # (6, )
"""

def split_x(seq, size):
    aaa = []        # 는 리스트
    for i in range(len(seq) - size + 1):
        subset = seq[i : (i+size)]
        aaa.append([item for item in subset])
    return np.array(aaa)

dataset = split_x(a, size)
print(dataset)
print(dataset.shape)                # 6, 5
# print(type(dataset))              # numpy.ndarray

# x,y 값 나누기
x = dataset[:, 0:4]                 # : 모든 행, 그 다음 0:4
y = dataset[:, 4]                   # : 모든 행, 인덱스 4부분만 가져오겠다.

print(x.shape)                      # 6, 4
print(y.shape)                      # 6,

x = x.reshape(x.shape[0], x.shape[1],1)
# 동일 문법
# x = np.reshape(x, (6,4,1))
# x = x.reshape(6, 4, 1)

print(x.shape)                      # 6, 4, 1


''' x, y 데이터 형태
        x           y
 [[[ 1  2  3  4 | 5]]

 [[ 2  3  4  5  6]]

 [[ 3  4  5  6  7]]

 [[ 4  5  6  7  8]]

 [[ 5  6  7  8  9]]

 [[ 6  7  8  9 10]]]

 # = 이렇게 되는 것
 x = np.array([[1,2,3,4],[2,3,4,5],[3,4,5,6],[4,5,6,7],[5,6,7,8],[6,7,8,9]])
 y = np.array([5,6,7,8,9,10])
 print("x_shape :", x.shape) # (6,4)
 print("y_shape :", y.shape) # (6, )

 x = x.reshape(x.shape[0], x.shape[1], 1) # (6,4,1)
 print(x.shape)
'''

# 2. 모델
model = Sequential()
# model.add(LSTM(300, input_shape=(4,1)))
model.add(LSTM(300, input_length=4, input_dim=1))
model.add(Dense(200))
model.add(Dense(150))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(1))

model.summary()


# EarlyStopping
from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss', patience=100, mode='auto')

# 3. 훈련
model.compile(loss='mse', optimizer='adam', metrics = ['mse'] )
model.fit(x, y, epochs=800, batch_size=32, verbose=2,
          callbacks=[es])
          # 여기서 batch_size 와 쉐이프의 batch_size 는 다르다. WHY ?
          # shape의 batch_size는 총 행의 수, fit의 batch_size는 거기서 n개씩 작업하겠다는 것.

# 4. 평가, 예측
loss, mse = model.evaluate(x, y, batch_size=32)

print('loss:', loss)
print('mse:', mse)

y_predict = model.predict(x)
print(y_predict)