# 200519 10:45 ~ < 하이퍼 파라미터 튜닝 : RMSE = 0.053, R2 = 0.999996 >

# 1. 데이터
import numpy as np
x1 = np.array([range(1,101), range(311, 411), range(411, 511)]) # (100,3)
x2 = np.array([range(711,811), range(711, 811), range(511, 611)])

y1 = np.array([range(101,201), range(411, 511), range(100)])

x1 = np.transpose(x1)
x2 = np.transpose(x2)
y1 = np.transpose(y1)

from sklearn.model_selection import train_test_split
x1_train, x1_test, x2_train, x2_test, y1_train, y1_test = train_test_split(
    x1, x2, y1, random_state = 66, shuffle = True,
    # x1, x2, y1, shuffle = False,
    train_size = 0.8)


#2. 모델구성
from keras.models import Sequential, Model 
from keras.layers import Dense, Input 

######### 모델 1 #########
input1 = Input(shape =(3, ))           
dense1_1 = Dense(6, activation='relu', name='dense1_1')(input1)
dense1_1 = Dense(9, activation='relu', name='dense1_2')(dense1_1)
dense1_1 = Dense(12, activation='relu', name='dense1_10')(dense1_1)
dense1_2 = Dense(15, activation = 'relu')(dense1_1)
   

######### 모델 2 #########
input2 = Input(shape =(3, )) 
dense2_1 = Dense(6, activation='relu', name='dense2_1')(input2)
dense2_1 = Dense(9, activation='relu', name='dense2_2')(dense2_1)
dense2_1 = Dense(12, activation='relu', name='dense2_10')(dense2_1)
dense2_2 = Dense(15, activation = 'relu')(dense2_1)
  

######### 모델 병합#########
from keras.layers.merge import concatenate   
merge1 = concatenate([dense1_2, dense2_2])   

middle1 = Dense(15)(merge1)
middle1 = Dense(20)(middle1) 
middle1 = Dense(25)(middle1) 
middle1 = Dense(30)(middle1) 
middle1 = Dense(35)(middle1) 


######### output 모델 구성 ###########
output1 = Dense(35)(middle1)   
output1_2 = Dense(40)(output1)
output1_2 = Dense(50)(output1_2)
output1_2 = Dense(30)(output1_2)
output1_2 = Dense(20)(output1_2)
output1_2 = Dense(10)(output1_2)
output1_2 = Dense(5)(output1_2)
output1_3 = Dense(3)(output1_2) 


######### 모델 명시 #########
model = Model(inputs = [input1, input2],
              outputs= output1_3) 

model.summary() 


# 3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit([x1_train, x2_train],
          y1_train, epochs=50, batch_size=2,
          validation_split=0.25, verbose=2)


# 4. 평가, 예측
loss = model.evaluate([x1_test, x2_test],
                      y1_test, batch_size=2)

print("model.metrics_names : ", model.metrics_names) 

print("loss :", loss)
# print("mse :", mse)

y1_predict = model.predict([x1_test, x2_test])
print(y1_predict)


# RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict)) # ()안은 매개변수니 건들 필요 없고

print("RMSE : ", RMSE(y1_test, y1_predict))


# R2 구하기
from sklearn.metrics import r2_score
r2 = r2_score(y1_test, y1_predict)
print("R2 : ", r2)