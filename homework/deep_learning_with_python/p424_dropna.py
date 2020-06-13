# p424~
# 14.3 결측치
# 14.3.1 리스트와이즈 삭제와 페어와이즈 삭제

import numpy as np
from numpy import nan as NA
import pandas as pd

sample_data_frame = pd.DataFrame(np.random.rand(10, 4))

# 일부 데이터 누락
sample_data_frame.iloc[1, 0] = NA
sample_data_frame.iloc[2, 2] = NA
sample_data_frame.iloc[5:, 3] = NA

# sample_data_frame
print(sample_data_frame)
#           0         1         2         3
# 0  0.221927  0.502284  0.738819  0.822059
# 1       NaN  0.823799  0.888840  0.483237
# 2  0.586563  0.093635       NaN  0.271853
# 3  0.952832  0.196862  0.909692  0.380966
# 4  0.918263  0.838170  0.507249  0.178438
# 5  0.267330  0.579828  0.789202       NaN
# 6  0.125356  0.251027  0.920030       NaN
# 7  0.969195  0.010895  0.741522       NaN
# 8  0.692045  0.632805  0.586715       NaN
# 9  0.714601  0.535878  0.560277       NaN

# NaN을 가진 행을 통째로 지우는 것 = listwise deletion

sdf_listwise = sample_data_frame.dropna()

print(sdf_listwise)
#           0         1         2         3
# 0  0.665054  0.814857  0.301137  0.675838
# 3  0.051849  0.043273  0.518985  0.160276
# 4  0.277398  0.006867  0.402619  0.877423

# 결손이 적은 열만 남기는 것 = pairwise deletion

sdf_pairwise = sample_data_frame[[0,1,2]].dropna()
print(sdf_pairwise)
#           0         1         2
# 0  0.448738  0.195867  0.094589
# 3  0.481421  0.995101  0.610625
# 4  0.868625  0.565186  0.178694
# 5  0.562685  0.229948  0.821367
# 6  0.959006  0.196997  0.062663
# 7  0.208001  0.848827  0.652114
# 8  0.802133  0.782535  0.141642
# 9  0.169514  0.662139  0.357052
