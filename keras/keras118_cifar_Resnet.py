# 20-07-03_28
# cifar10 과 ResNet 모델 엮기


from keras.datasets import cifar10
from keras.applications import VGG16, VGG19, Xception, ResNet101, ResNet101V2, ResNet152
from keras.applications import ResNet152V2, ResNet50, ResNet50V2, InceptionV3, InceptionResNetV2
from keras.applications import MobileNet, MobileNetV2, DenseNet121, DenseNet169, DenseNet201
from keras.applications import NASNetLarge, NASNetMobile

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, MaxPool2D, Flatten, BatchNormalization, Activation
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam

#1. data
(x_train, y_train),(x_test, y_test) = cifar10.load_data()

# 전처리
x_train = x_train.reshape(50000, 32, 32, 3).astype('float32')/255
x_test = x_test.reshape(10000, 32, 32, 3).astype('float32')/255


#2. model
rn50 = ResNet50(include_top = False, input_shape = (32, 32, 3))

# vgg.summary()

model = Sequential()
model.add(rn50)
model.add(Flatten())
model.add(Dense(10, activation = 'softmax'))

model.summary()


#3. compile, fit
from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='acc', patience=5)

model.compile(optimizer = Adam(1e-4), loss = 'sparse_categorical_crossentropy', metrics = ['acc'])                             
hist = model.fit(x_train, y_train, epochs = 10, batch_size = 32, verbose = 1, 
                 validation_split =0.3 , shuffle = True, callbacks=[es])


#4. evaluate
loss_acc = model.evaluate(x_test, y_test, batch_size = 32)

loss = hist.history['loss']
val_loss = hist.history['val_loss']
acc = hist.history['acc']
val_acc = hist.history['val_acc']

# print('acc 는 ', acc)
# print('val_acc 는 ', val_acc)

# evaluate 종속 결과
print('loss, acc 는 ', loss_acc)


##### plt 시각화
import matplotlib.pyplot as plt

plt.figure(figsize=(9,5))

# 1
plt.subplot(2,1,1)
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid()
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')

# 2
plt.subplot(2,1,2)
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.grid()
plt.title('acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['acc', 'val_acc'])

plt.show()

# loss, acc 는  [0.8258684811830521, 0.7906000018119812]