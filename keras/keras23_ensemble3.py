# 200518 16:45 ~ <x 데이터 2개, y 데이터 1개>

# 1. 데이터
import numpy as np
x1 = np.array([range(1,101), range(311, 411), range(411, 511)]) # (100,3)
x2 = np.array([range(711,811), range(711, 811), range(511, 611)])

y1 = np.array([range(101,201), range(411, 511), range(100)])


##### 수정 시작 ######

x1 = np.transpose(x1)
x2 = np.transpose(x2)
y1 = np.transpose(y1)


from sklearn.model_selection import train_test_split
x1_train, x1_test, x2_train, x2_test, y1_train, y1_test = train_test_split(
    x1, x2, y1, shuffle = False,
    train_size = 0.8)

# print(x1_train)
# print(x2_train)
# print(y1_train)

#### 이 방식이 선생님이 알려주신 원래 방법 ####
# from sklearn.model_selection import train_test_split
# x1_train, x1_test, y1_train, y1_test = train_test_split(
#     x1, y1, shuffle = False,
#     train_size = 0.8)

# x2_train, x2_test = train_test_split(
#     x2, shuffle = False,
#     train_size = 0.8)

#2. 모델구성
from keras.models import Sequential, Model 
from keras.layers import Dense, Input 

######### 모델 1 #########
input1 = Input(shape =(3, ))           
dense1_1 = Dense(11, activation = 'relu')(input1)
dense1_1 = Dense(7, activation = 'relu')(dense1_1)
dense1_2 = Dense(5, activation = 'relu')(dense1_1)
   

######### 모델 2 #########
input2 = Input(shape =(3, )) 
dense2_1 = Dense(11, activation = 'relu')(input2) 
dense2_1 = Dense(7, activation = 'relu')(dense2_1)
dense2_2 = Dense(5, activation = 'relu')(dense2_1)
  

######### 모델 병합#########
from keras.layers.merge import concatenate   
merge1 = concatenate([dense1_2, dense2_2])   

middle1 = Dense(9)(merge1)
middle1 = Dense(5)(middle1) 
middle1 = Dense(5)(middle1) 

######### output 모델 구성 ###########
output1 = Dense(5)(middle1)   
output1_2 = Dense(5)(output1)
output1_3 = Dense(3)(output1_2) 


######### 모델 명시 #########
model = Model(inputs = [input1, input2],
              outputs= output1_3) 

model.summary() 


# 3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit([x1_train, x2_train],
          y1_train, epochs=10, batch_size=1,
          validation_split=0.25, verbose=2)


# 4. 평가, 예측
loss = model.evaluate([x1_test, x2_test],
                      y1_test, batch_size=1)

print("model.metrics_names : ", model.metrics_names) 

print("loss :", loss)
# print("mse :", mse)

y1_predict = model.predict([x1_test, x2_test])
print(y1_predict)


# RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

print("RMSE : ", RMSE(y1_test, y1_predict))


# R2 구하기
from sklearn.metrics import r2_score
r2 = r2_score(y1_test, y1_predict)
print("R2 : ", r2)