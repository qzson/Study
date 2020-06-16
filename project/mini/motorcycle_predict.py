# Load 모델 (modelcheckpoint) & Predict

from PIL import Image
import os, glob, numpy as np
from keras.models import load_model

### predict 이미지 불러오기
caltech_dir = './project/mini/images/pred'

image_w = 100
image_h = 100

### pred 이미지를 Data 변환
X = []
filenames = []

files = glob.glob(caltech_dir + '/*.*')
for i, f in enumerate(files):
    img = Image.open(f)
    img = img.convert('RGB')
    img = img.resize((image_w, image_h))
    data = np.asarray(img)
    filenames.append(f)
    X.append(data)

x_pred = np.array(X)

### modelcheckpint Load
model = load_model('./project/mini/checkpoint/mcp-22-0.0638.hdf5')

### 예측
y_pred = model.predict(x_pred)
np.set_printoptions(formatter={'float': lambda x: '{0:0.2f}'.format(x)})
cn = 0

for i in y_pred:
    pre_idx = i.argmax() # 예측 레이블
    # print(i)
    # print(pre_idx)
    pre_idx_str = ''
    if pre_idx == 0: pre_idx_str = '( 스쿠터 )'
    elif pre_idx == 1: pre_idx_str = '( 수퍼스포츠 )'
    elif pre_idx == 2: pre_idx_str = '( 멀티퍼포스 )'
    else: pre_idx_str = '( 크루저 )'
    if i[0] >= 0.5 : print(filenames[cn].split('\\')[1] + ' 의 모델은 ' + pre_idx_str + ' 로 예측됩니다.')
    if i[1] >= 0.5 : print(filenames[cn].split('\\')[1] + ' 의 모델은 ' + pre_idx_str + ' 로 예측됩니다.')
    if i[2] >= 0.5 : print(filenames[cn].split('\\')[1] + ' 의 모델은 ' + pre_idx_str + ' 로 예측됩니다.')
    if i[3] >= 0.5 : print(filenames[cn].split('\\')[1] + ' 의 모델은 ' + pre_idx_str + ' 로 예측됩니다.')
    cn += 1
