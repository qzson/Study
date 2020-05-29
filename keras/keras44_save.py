# 200525 1400~
# save & load

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM

# 2. 모델
model = Sequential()
model.add(LSTM(200, input_shape=(4,1)))
model.add(Dense(150))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(25))
model.add(Dense(10))

model.summary()

""" 모델 SAVE 방법 """

model.save("./model/save_keras44.h5")         # // , / , \ : 하단 폴더
# model.save(".//model//save_keras44.h5")     #  . : 현재 폴더
# model.save(".\model\save_keras44.h5")    

# model.save("경로 / 파일 이름 .h5")

# 파라미터 튜닝 했는데, 잘 나온 값 저장해 놓을 수 있다.
print("저장 잘 됐다.")