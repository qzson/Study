# 다중 분류 모델 1000~, 1100~
# loss, Output 변경. (*Output 변경 시, y값도 변경 해야한다.)
# train test 분리 X
# 모델을 짜라

''' 과제2
 y_pred 값을 숫자로 만들어주는 함수 '''

import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# 1. 데이터 (이번엔 다중분류 : 5진 분류)
x = np.array(range(1, 11))
y = np.array([1,2,3,4,5,1,2,3,4,5])
 # 분류를 5개 주었다. (1->1, 2->2...6 ->1, 7->2,...10->5)

from keras.utils import np_utils
 # 데이터가 공평하게 빠질 수 있도록 2차원으로 바꿔준다.
 # (원핫 인코딩) y값이 2.5배가 되는 그런 현상이 나타나지 않게 하기 위해서
y = np_utils.to_categorical(y)
 # 다중 분류 시 필수로 원핫 인코딩을 해야한다.

print(x.shape)
print(y.shape)                      # (10,6)
print(y)

y = y[:, 1:]
print(y)                            # (10,5)

''' one hot encording
 #      1,2,3,4,5 그 순서에 맞는 것만 1을 부여
 # 즉,  1 0 0 0 0
 #      0 1 0 0 0 ...
 # 그 자리에 1이 들어감으로 스칼라가 2차원형태로 된다.
 10, 6으로 나와있는데 원래는 10, 5 로 만들어 줘야한다. (실제적으로 원핫인코딩시 5행이 나와야하니까)
 
 - 다중 분류 모델은 반드시 one_hot_encoding사용
 - 다중 클래스 분류 문제가 각 클래스 간의 관계가 균등하기 때문에
 ex) y가 1 과 5로 분류된다면 5에 값이 치중된다. '''


# 2. 모델
model = Sequential()
model.add(Dense(200,activation = 'relu', input_dim = 1))
model.add(Dense(150,activation = 'relu'))
model.add(Dense(120,activation = 'relu'))
model.add(Dense(90,activation = 'relu'))
model.add(Dense(60,activation = 'relu'))
model.add(Dense(40,activation = 'relu'))
model.add(Dense(30,activation = 'relu'))
model.add(Dense(20,activation = 'relu'))
model.add(Dense(10,activation = 'relu'))        
model.add(Dense(5,activation = 'softmax'))
""" 다중 분류는 'softmax' 사용
   : 가장 큰 수 빼고는 전부 0으로 나옴 """


# 3. 훈련
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
 # 2진 분류 : binary_crossentropy / 다중 분류 : categorical_crossentropy
model.fit(x, y, epochs=90, batch_size=1, verbose=2)


# 4. 평가, 예측
loss, acc = model.evaluate(x, y, batch_size=1)
print("loss :", loss)
print("acc :", acc)

x_pred = np.array([1,2,3])
y_pred = model.predict(x_pred)                     # 'softmax'가 적용되지 않은 모습으로 나옴
# y_pred = np.argmax(y_pred, axis=1)+1
print(y_pred)
print(y_pred.shape)                                # (3, 5)

""" one_hot_decoder
 : np.argmax()
 : 최대값의 색인 위치를 찾는다. """

''' 출력 (10,6 기준)

 1. [[0.03578157 0.20864153 0.27041987 0.18988359 0.15629996 0.1389734 ]        
 2. [0.02184891 0.19430807 0.25464013 0.19235192 0.17690909 0.15994194]        
 3. [0.01358252 0.17784293 0.23612909 0.19247909 0.19741626 0.1825501 ]]
 1,2,3 의 각각 6개
 
 출력 (10, 5) 기준
 [[0.23078643 0.1998456  0.24431306 0.17311464 0.15194027]
 [0.22345239 0.19755079 0.23983486 0.1819732  0.15718876]
 [0.20341207 0.19506593 0.22195956 0.19830263 0.1812599 ]].
 

 가장 큰 값이 숫자 부여 인덱스
 1. 0, 1, 2, 3, 4 ,5 = 2
 2. = 2
 3. = 2
 
 과제: 결과값을 숫자로 바꿔주는 함수가 있다. '''

# 1. 함수 사용
'''
def decode(datum):
    return np.argmax(datum)
  
for i in range(y_pred.shape[0]):                   # y_pred.shape[0] = 3, i = [0, 1, 2]                     
    y2_pred = decode( y_pred[i])       
    print('y2_pred:', y2_pred + 1) '''

#2. argmax 사용
y3_pred = np.argmax(y_pred, axis= 1) + 1           # 뒤로 한자리씩 넘겨준다.
print(y3_pred)