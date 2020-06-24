import numpy as np
import pandas as pd
import pywt
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler

def outliers(data, axis= 0):
    if type(data) == pd.DataFrame:
        data = data.values
    if len(data.shape) == 1:
        quartile_1, quartile_3 = np.percentile(data,[25,75])
        print("1사분위 : ", quartile_1)
        print("3사분위 : ", quartile_3)
        iqr = quartile_3 - quartile_1
        lower_bound = quartile_1 - (iqr * 1.5)  ## 아래
        upper_bound = quartile_3 + (iqr * 1.5)  ## 위
        return np.where((data > upper_bound) | (data < lower_bound))
    else:
        output = []
        for i in range(data.shape[axis]):
            if axis == 0:
                quartile_1, quartile_3 = np.percentile(data[i, :],[25,75])
            else:
                quartile_1, quartile_3 = np.percentile(data[:, i],[25,75])
            iqr = quartile_3 - quartile_1
            lower_bound = quartile_1 - (iqr * 1.5)  ## 아래
            upper_bound = quartile_3 + (iqr * 1.5)  ## 위
            if axis == 0:
                output.append(np.where((data[i, :] > upper_bound) | (data[i, :] < lower_bound))[0])
            else:
                output.append(np.where((data[:, i] > upper_bound) | (data[:, i] < lower_bound))[0])
    return np.array(output)


## 데이터 불러오기
train = pd.read_csv('./data/dacon/comp1/train.csv', sep=',', index_col = 0, header = 0)
test = pd.read_csv('./data/dacon/comp1/test.csv', sep=',', index_col = 0, header = 0)


## 데이터 분해 및 columns이름 저장
train_col = train.columns[:-4]
test_col = test.columns
y_train_col = train.columns[-4:]
y_train = train.values[:,-4:]
train = train.values[:,:-4]
test = test.values

# scaler = MinMaxScaler()
# train = scaler.fit_transform(train)
# test = scaler.transform(test)


## NaN값이 있는 train,test값을 다시 데이터 프레임으로 감싸주기
train = pd.DataFrame(train, columns=train_col)
test = pd.DataFrame(test, columns=test_col)

# tmp = outliers(test,1)
# print(tmp.shape)
# temp = []
# for i in range(tmp.shape[0]):
#     temp += list(tmp[i])
# print(len(list(set(temp))))

# tmp = outliers(train,1)
# print(tmp.shape)
# temp = []
# for i in range(tmp.shape[0]):
#     temp += list(tmp[i])
# print(len(list(set(temp))))

train_src = train.filter(regex='_src$',axis=1).T.interpolate(limit_direction='both').T.values # 선형보간법
train_dst = train.filter(regex='_dst$',axis=1).T.interpolate(limit_direction='both').T.values * (10 ** (train.values[:,0:1] * train.values[:,0:1]/100)) # 선형보간법
test_src = test.filter(regex='_src$',axis=1).T.interpolate(limit_direction='both').T.values
test_dst = test.filter(regex='_dst$',axis=1).T.interpolate(limit_direction='both').T.values * (10 ** (test.values[:,0:1] * test.values[:,0:1]/100))
print(train_dst)

# for i in range(10):
#     plt.subplot(2,2,1)
#     plt.plot(range(35), train_src[i])
#     plt.subplot(2,2,2)
#     plt.plot(range(35), train_dst[i])
#     plt.subplot(2,2,3)
#     plt.plot(range(35), test_src[i])
#     plt.subplot(2,2,4)
#     plt.plot(range(35), test_dst[i])
#     plt.show()


# (ca, cd) = pywt.dwt(train_src, "haar")
# cat = pywt.threshold(ca, np.std(ca), mode="hard")
# cdt = pywt.threshold(cd, np.std(cd), mode="hard")
# train_src = pywt.idwt(cat, cdt, "haar")[:,:35]

# scaler = MinMaxScaler()
# train_dst = scaler.fit_transform(train_dst.T).T
# (ca, cd) = pywt.dwt(train_dst, "haar")
# cat = pywt.threshold(ca, np.std(ca), mode="hard")
# cdt = pywt.threshold(cd, np.std(cd), mode="hard")
# train_dst = pywt.idwt(cat, cdt, "haar")
# train_dst = scaler.inverse_transform(train_dst[:,:35].T).T

# (ca, cd) = pywt.dwt(test_src, "haar")
# cat = pywt.threshold(ca, np.std(ca), mode="hard")
# cdt = pywt.threshold(cd, np.std(cd), mode="hard")
# test_src = pywt.idwt(cat, cdt, "haar")[:,:35]

# test_dst = scaler.fit_transform(test_dst.T).T
# (ca, cd) = pywt.dwt(test_dst, "haar")
# cat = pywt.threshold(ca, np.std(ca), mode="hard")
# cdt = pywt.threshold(cd, np.std(cd), mode="hard")
# test_dst = pywt.idwt(cat, cdt, "haar")
# test_dst = scaler.inverse_transform(test_dst[:,:35].T).T

# for i in range(10):
#     plt.subplot(2,2,1)
#     plt.plot(range(36), train_src[i])
#     plt.subplot(2,2,2)
#     plt.plot(range(35), train_dst[i])
#     plt.subplot(2,2,3)
#     plt.plot(range(36), test_src[i])
#     plt.subplot(2,2,4)
#     plt.plot(range(35), test_dst[i])
#     plt.show()


max_test = 0
max_train = 0
train_fu_real = []
train_fu_imag = []
test_fu_real = []
test_fu_imag = []

for i in range(10000):
    tmp_x = 0
    tmp_y = 0
    for j in range(35):
        # if train_src[i, j] == 0:
        #     tmp_x += 1
        #     train_src[i,j] = 0
        #     train_dst[i,j] = 0
        if train_src[i, j] - train_dst[i, j] < 0:
            train_src[i,j] = train_dst[i,j] 
        # if test_src[i, j] == 0:
        #     tmp_y += 1
        #     test_src[i,j] = 0
        #     test_dst[i,j] = 0
        if test_src[i, j] - test_dst[i, j] < 0:
            test_src[i,j] = test_dst[i,j]
    if tmp_x > max_train:
        max_train = tmp_x
    if tmp_y > max_test:
        max_test = tmp_y
    train_fu_real.append(np.fft.fft(train_dst[i]-train_dst[i].mean()).real)
    train_fu_imag.append(np.fft.fft(train_dst[i]-train_dst[i].mean()).imag)
    test_fu_real.append(np.fft.fft(test_dst[i]-test_dst[i].mean()).real)
    test_fu_imag.append(np.fft.fft(test_dst[i]-test_dst[i].mean()).imag)
print(max_train)
print(max_test)
# print(train_fu_real)
# print(train_fu_imag)
# print(test_fu_real)
# print(test_fu_imag)

# max_test = 0
# max_train = 0
# train_fu_real = []
# train_fu_imag = []
# test_fu_real = []
# test_fu_imag = []

# for i in range(10000):
#     tmp_x = 0
#     tmp_y = 0
#     for j in range(35):
#         if train.iloc[i, j+1] == 0:
#             tmp_x += 1
#             train.iloc[i,j+1] = 0
#             train.iloc[i,j+36] = 0
#         if train.iloc[i, j+1] - train.iloc[i, j+36] < 0:
#             train.iloc[i,j+36] = train.iloc[i,j+1]
#         if test.iloc[i, j+1] == 0:
#             tmp_y += 1
#             test.iloc[i,j+1] = 0
#             test.iloc[i,j+36] = 0
#         if test.iloc[i, j+1] - test.iloc[i, j+36] < 0:
#             test.iloc[i,j+36] =test.iloc[i,j+1] 
#     if tmp_x > max_train:
#         max_train = tmp_x
#     if tmp_y > max_test:
#         max_test = tmp_y
#     train_fu_real.append(np.fft.ifft(train.iloc[i, 36:71]-train.iloc[i, 36:71].mean(), norm='ortho').real)
#     train_fu_imag.append(np.fft.ifft(train.iloc[i, 36:71]-train.iloc[i, 36:71].mean(), norm='ortho').imag)
#     test_fu_real.append(np.fft.ifft(test.iloc[i, 36:71]-test.iloc[i, 36:71].mean(), norm='ortho').real)
#     test_fu_imag.append(np.fft.ifft(test.iloc[i, 36:71]-test.iloc[i, 36:71].mean(), norm='ortho').imag)
# print(max_train)
# print(max_test)
# print(train_fu_real)
# print(train_fu_imag)
# print(test_fu_real)
# print(test_fu_imag)

# print(train.isnull().sum())
# train.to_csv('./example.csv',index=False)
# # print(train)

# train에서 
# train_src = train.filter(regex='_src$',axis=1).values#.T.interpolate().fillna(method ='ffill').fillna(method ='bfill').T.values # 선형보간법
# train_dst = train.filter(regex='_dst$',axis=1).values * train.values[:,0:1] * train.values[:,0:1]#.T.interpolate().fillna(method ='ffill').fillna(method ='bfill').T.values # 선형보간법
# test_src = test.filter(regex='_src$',axis=1).values#.T.interpolate().fillna(method ='ffill').fillna(method ='bfill').T.values
# test_dst = test.filter(regex='_dst$',axis=1).values* test.values[:,0:1]* test.values[:,0:1]#.T.interpolate().fillna(method ='ffill').fillna(method ='bfill').T.values

print(((train_src - train_dst) < 0).sum())
print(((test_src - test_dst) < 0).sum())

# train_src_rank = np.argsort(train_src)[::-1][:, :10]
# train_dst_rank = np.argsort(train_dst)[::-1][:, :10]
# test_src_rank = np.argsort(test_src)[::-1][:, :10]
# test_dst_rank = np.argsort(test_dst)[::-1][:, :10]
# print(np.array(train_src - train_dst)[:2,:])






small = 1e-30

# x_train = np.concatenate([train.values[:,0:1]**2, train_src/(train.values[:,0:1]**2), train_dst, train_src/(train.values[:,0:1]**2) - train_dst,train_src/(train.values[:,0:1]**2)/(train_dst+small)], axis = 1)
# x_pred = np.concatenate([test.values[:,0:1]**2, test_src/(train.values[:,0:1]**2), test_dst, test_src/(train.values[:,0:1]**2) - test_dst,test_src/(train.values[:,0:1]**2)/(test_dst+small)], axis = 1)


x_train = np.concatenate([train.values[:,0:1]**2,train_dst, train_src-train_dst, train_src/(train_dst+small), train_fu_real, train_fu_imag] , axis = 1)
x_pred = np.concatenate([test.values[:,0:1]**2,test_dst, test_src-test_dst, test_src/(test_dst+small), test_fu_real, test_fu_imag], axis = 1)

# x_train = np.concatenate([train.values[:,0:1]**2, train_src, train_dst, train_src - train_dst,train_src/(train_dst+small)], axis = 1)
# x_pred = np.concatenate([test.values[:,0:1]**2, test_src, test_dst, test_src - test_dst,test_src/(test_dst+small)], axis = 1)

# x_train = np.concatenate([train.values[:,0:1]**2, train_src, train_dst, train_src - train_dst,train_src/(train_dst+small), np.log10((train_src + small)/(train_dst+small))], axis = 1)
# x_pred = np.concatenate([test.values[:,0:1]**2, test_src, train_dst, test_src - test_dst,test_src/(test_dst+small), np.log10((test_src+small)/(test_dst+small))], axis = 1)
# print(pd.DataFrame(x_train).isnull().sum())
# print(pd.DataFrame(np.log10(train_src.values) - np.log10(train_dst.values)))

print(x_train.shape)
print(y_train.shape)
print(x_pred.shape)

np.save('./data/dacon/comp1/x_train.npy', arr=x_train)
np.save('./data/dacon/comp1/y_train.npy', arr=y_train)
np.save('./data/dacon/comp1/x_pred.npy', arr=x_pred)

## 푸리에 변환 
# 섞여버린 파형을 여러개으 순수한 음파로 분해하는 방법

## 이미 이 데이터는 푸리에 변환이 되어있는 데이터이다?
## 각 파장의 세기 그래프 == 푸리에변환
## 분광분석법 
## IQR = 4분할
## 각 IQR*1.5의 지점을 벗어나는 값을 이상치라고 한다.