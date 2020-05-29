import numpy as np
from sklearn.datasets import load_diabetes

dataset = load_diabetes()
x = dataset.data
y = dataset.target

print(x.shape) # (442, 10)
print(y.shape) # (442, )

# 2. 모델
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(100,activation = 'relu', input_dim = 10))
model.add(Dense(90,activation = 'relu'))
model.add(Dense(70,activation = 'relu'))
model.add(Dense(60,activation = 'relu'))
model.add(Dense(50,activation = 'relu'))
model.add(Dense(40,activation = 'relu'))
model.add(Dense(30,activation = 'relu'))
model.add(Dense(20,activation = 'relu'))
model.add(Dense(10,activation = 'relu'))     
model.add(Dense(1,activation = 'sigmoid'))


# 3. 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x, y, epochs=500, batch_size=32, verbose=2)
 

# 4. 평가, 예측
loss, acc = model.evaluate(x, y, batch_size=32)
print("loss :", loss)
print("acc :", acc)

# x_pred = np.array([1,2,3])
# y_pred = model.predict(x_pred)
# print(y_pred)
# # sigmoid 함수를 거치지 않은 걸로 보여짐

# y1_pred = np.where(y_pred >= 0.5, 1, 0)     
# print('y_pred :', y1_pred)