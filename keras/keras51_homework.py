# 20.05.27_13일차 / 0900~
# 2번의 첫번째 답

'''
 # x = [1,2,3]
 # x = x -1
 # print(x)
  # TypeError: unsupported operand type(s) for -: 'list' and 'int'
  # 그래서 numpy를 쓴다. 하지만 numpy 단점은 자료형은 두 가지 이상 사용할 수 없다. '''


import numpy as np
y = np.array([1,2,3,4,5,1,2,3,4,5])
y = y -1

# print(y)                                    # [0 1 2 3 4 0 1 2 3 4]
 # 이런 방법도 있다.
 # x값과 y값이 매치가 되어있는 상태기 때문에 순서만 잘 맞춰주면 x,y가 서로 틀어지지 않는다.
 # 그런데, y에서 -1을 왜 빼주는 것인가 연구의 필요성 있음


from keras.utils import np_utils
y = np_utils.to_categorical(y)
print(y)
print(y.shape)

'''
# 2번의 두번째 답
# y = np.array([1,2,3,4,5,1,2,3,4,5])
print(y.shape) # (10, )
               # sklearn은 2차원 형식으로 들어가야해서 차원을 바꿔줘야 한다.
y = y.reshape(-1, 1)                        # -1 인덱스의 제일 마지막
# y = y.reshape(10, 1) [같은 문법]


from sklearn.preprocessing import OneHotEncoder
aaa = OneHotEncoder()
aaa.fit(y)
y = aaa.transform(y).toarray()
 # 얘는 숫자만큼 된다. 그렇지만 단점이 있다? (질문) 무슨 단점이 있나요?
print(y)
print(y.shape)'''
