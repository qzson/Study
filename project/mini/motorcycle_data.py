# 이미지 데이터 처리

from PIL import Image
import os, glob, numpy as np
from sklearn.model_selection import train_test_split


### 이미지 파일 불러오기 및 카테고리 정의
caltech_dir = './project/mini/images'
categories = ['scooter', 'supersports', 'multipurpose', 'cruiser']
nb_classes = len(categories)

### 가로, 세로, 채널 쉐이프 정의
image_w = 100
image_h = 100

### 이미지 파일 Data화
X = []
Y = []

for idx, cate in enumerate(categories):
    label = [0 for i in range(nb_classes)]
    label[idx] = 1
    image_dir = caltech_dir + '/' + cate
    files = glob.glob(image_dir + "/*.jpg")
    print(cate, " 파일 길이 : ", len(files))
    for i, f in enumerate(files):
        img = Image.open(f)
        img = img.convert('RGB')
        img = img.resize((image_w, image_h))
        data = np.asarray(img)

        X.append(data)
        Y.append(label)

        if i % 49 == 0:
           print(cate, ':', f)

x = np.array(X)
y = np.array(Y)

print(x.shape) # (200, 100, 100, 3)
print(y.shape) # (200, 4)

# ### 데이터 train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=66, shuffle=True)
xy = (x_train, x_test, y_train, y_test)
# # print(x_train.shape)    # (160, 100, 100, 3)
# # print(x_test.shape)     # (40, 100, 100, 3)
# # print(y_train.shape)    # (160, 4)
# # print(y_test.shape)     # (40, 4)

### 데이터 SAVE
np.save('./project/mini/data/multi_image_data.npy', xy)
print('ok', len(y))