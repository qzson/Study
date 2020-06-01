# 20-06-01 / 월 / 15:00~
# CSV 파일 가져오기 k91 copy

''' numpy save와 CSV 장난질 
 넘파이는 한가지 자료형에만 사용 가능하다.
 pandas는 자료형에 유연하다.
 '''

from sklearn.datasets import load_iris
import numpy as np

iris = load_iris()

print(type(iris))   # <class 'sklearn.utils.Bunch'> : sklearn에서 제공하는 파일
x_data = iris.data
y_data = iris.target

print(type(x_data)) # numpy.ndarray
print(type(y_data)) # numpy.ndarray

np.save('./data/iris_x.npy', arr=x_data) # np로 데이터 저장
np.save('./data/iris_y.npy', arr=y_data)

x_data_load = np.load('./data/iris_x.npy') # np로 저장된 데이터 로드
y_data_load = np.load('./data/iris_y.npy')

print(type(x_data_load)) # numpy.ndarray
print(type(y_data_load)) # numpy.ndarray
print(x_data_load.shape) # (150, 4)
print(y_data_load.shape) # (150, )