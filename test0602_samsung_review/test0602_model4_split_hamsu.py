# 20-06-03 자습


import numpy as np

# dataset.npy 불러오기

samsung = np.load('./data/samsung_test.npy', allow_pickle='True')
hite = np.load('./data/hite_test.npy', allow_pickle='True')
print(samsung.shape)                                              # (509, 1)
print(hite.shape)                                                 # (509, 5)
# print(samsung)
# print(hite)

# print(samsung[0:6], '\n\n', samsung[1:7])
# print('\n')
# print(samsung[502:508], '\n\n', samsung[503:509], '\n\n', samsung[504:510])
                                                    # y 마지막 값 비교해보자
''' print(samsung[0:6] ~ [504:510])
 [[53000]
 [52600]
 [52600]
 [51700]
 [52000]
 [51000]]

 [[52600]
 [52600]
 [51700]
 [52000]
 [51000]
 [50200]]


 [[48750]
 [48700]
 [48950]
 [51100]
 [50000]
 [50800]]

 [[48700]
 [48950]
 [51100]
 [50000]
 [50800]
 [51000]]

 [[48950]
 [51100]
 [50000]
 [50800]
 [51000]]
 '''

# print(hite[0:6], '\n\n', hite[1:7])
# print('\n')
# print(hite[502:508], '\n\n', hite[503:509], '\n\n', hite[504:510])

''' print(hite[0:6] ~ hite[504:510])
 [[21400 21600 21350 21550 123592]
 [21450 21550 21000 21050 250520]
 [21050 21200 20950 21100 165195]
 [21150 21250 20900 21000 215315]
 [21000 21550 21000 21400 229912]
 [21400 21800 21400 21400 281871]]

 [[21450 21550 21000 21050 250520]
 [21050 21200 20950 21100 165195]
 [21150 21250 20900 21000 215315]
 [21000 21550 21000 21400 229912]
 [21400 21800 21400 21400 281871]
 [21400 21700 21400 21550 87731]]


 [[36000 36500 35700 36400 422582]
 [36450 36500 35750 36100 409419]
 [35900 36450 35800 36400 373464]
 [36200 36300 35500 35800 548493]
 [35900 36750 35900 36000 576566]
 [36000 38750 36000 38750 1407345]]

 [[36450 36500 35750 36100 409419]
 [35900 36450 35800 36400 373464]
 [36200 36300 35500 35800 548493]
 [35900 36750 35900 36000 576566]
 [36000 38750 36000 38750 1407345]
 [39000 38750 36000 38750 1407345]]

 [[35900 36450 35800 36400 373464]
 [36200 36300 35500 35800 548493]
 [35900 36750 35900 36000 576566]
 [36000 38750 36000 38750 1407345]
 [39000 38750 36000 38750 1407345]]
 '''


''' 스플릿 함수 정의 '''

def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1):    # for i in range(509 - 6 + 1) = for i in range(504)
        subset = seq[i:(i+size)]            # subset = seq[0:(0+6)] = seq[0:6] = (인덱스 개념) [0,1,2,3,4,5]
        aaa.append([j for j in subset])     # aaa[i] = aaa[0] = [0,1,2,3,4,5], i=1 => [1,2,3,4,5,6]
    return np.array(aaa)                    # ([0~5], [1~6], [2~7], ... , [504~509])
size = 6            # 6일치씩 자르겠다

samsung = samsung.reshape(samsung.shape[0],)# (509,)
samsung = split_x(samsung, size)
print(samsung.shape)                        # (504, 6)
print(samsung)