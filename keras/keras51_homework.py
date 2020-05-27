# 20.05.27_13일차 / 0900~
# 2번의 첫번째 답

# keras49_classify2.py
# ome_hot_encoding
# 무조건 인덱스의 시작이 0부터 시작된다.
# 1.앞에 0을 자르는 방법
# - 슬라이싱사용 
# - sk.learn의 one_hot_encoding사용
# - 판다스 사용

'''
 # x = [1,2,3]
 # x = x -1
 # print(x)
  # TypeError: unsupported operand type(s) for -: 'list' and 'int'
  # 그래서 numpy를 쓴다. 하지만 numpy 단점은 자료형은 두 가지 이상 사용할 수 없다. '''

# 과제2 1번째 답
import numpy as np
y = np.array([1,2,3,4,5,1,2,3,4,5])
y = y - 1                                     # numpy에서만 가능
                                              # 단점 : 한가지 자료형만 써야 한다.
print(y)                                      # [0 1 2 3 4 0 1 2 3 4]
 # 이런 방법도 있다.
 # x값과 y값이 매치가 되어있는 상태기 때문에 순서만 잘 맞춰주면 x,y가 서로 틀어지지 않는다.


from keras.utils import np_utils              # shape가 0부터 시작이 된다.
y = np_utils.to_categorical(y)
print(y)
print(y.shape)                                # (10, 5)
"""
[[1. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0.]
 [0. 0. 1. 0. 0.]
 [0. 0. 0. 1. 0.]
 [0. 0. 0. 0. 1.]
 [1. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0.]
 [0. 0. 1. 0. 0.]
 [0. 0. 0. 1. 0.]
 [0. 0. 0. 0. 1.]]
 (10, 5) """  


# 과제2 2번째 답

y1 = np.array([1,2,3,4,5,1,2,3,4,5])
print(y1.shape)                             # (10, )
                                            # sklearn은 2차원 형식으로 들어가야해서 차원을 바꿔줘야 한다.
                                            # sklearn의 OneHotEncoder을 사용하기 위해서는 2차원이어야 한다.
y1 = y1.reshape(-1, 1)                      # -1 인덱스의 제일 마지막
# y1 = y1.reshape(10, 1) [같은 문법]


from sklearn.preprocessing import OneHotEncoder
aaa = OneHotEncoder()
aaa.fit(y1)
y1 = aaa.transform(y1).toarray()            # sklearn은 그 숫자만큼 생성
 # 그렇지만 단점이 있다? (질문) 무슨 단점이 있나요?
print(y1)
print(y1.shape)
'''
 [[1. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0.]
 [0. 0. 1. 0. 0.]
 [0. 0. 0. 1. 0.]
 [0. 0. 0. 0. 1.]
 [1. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0.]
 [0. 0. 1. 0. 0.]
 [0. 0. 0. 1. 0.]
 [0. 0. 0. 0. 1.]]
(10, 5) '''