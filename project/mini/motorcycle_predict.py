# Load 모델 & Predict

from PIL import Image
import os, glob, numpy as np
from keras.models import load_model

caltech_dir = './project/mini/images/pred'

image_w = 100
image_h = 100

pixels = image_w * image_h * 3

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
model = load_model('./project/mini/checkpoint/cp-10-0.0935.hdf5')

y_pred = model.predict(x_pred)
np.set_printoptions(formatter={'float': lambda x: '{0:0.3f}'.format(x)})
cnt = 0

for i in y_pred:
    pre_ans = i.argmax() # 예측 레이블
    print(i)
    print(pre_ans)
    pre_ans_str = ''
    if pre_ans == 0: pre_ans_str = '스쿠터'
    elif pre_ans == 1: pre_ans_str = '수퍼스포츠'
    elif pre_ans == 2: pre_ans_str = '멀티퍼포스'
    else: pre_ans_str = '크루저'
    if i[0] >= 0.8 : print('해당' + filenames[cnt].split('\\')[1] + '이미지는' + pre_ans_str + '로 추정됩니다.')
    if i[1] >= 0.8 : print('해당' + filenames[cnt].split('\\')[1] + '이미지는' + pre_ans_str + '로 추정됩니다.')
    if i[2] >= 0.8 : print('해당' + filenames[cnt].split('\\')[1] + '이미지는' + pre_ans_str + '로 추정됩니다.')
    if i[3] >= 0.8 : print('해당' + filenames[cnt].split('\\')[1] + '이미지는' + pre_ans_str + '로 추정됩니다.')
    cnt += 1
