from PIL import Image
import os, glob, numpy as np
from sklearn.model_selection import train_test_split

caltech_dir = './project/mini/images'
categories = ['scooter', 'supersports', 'multipurpose', 'cruiser']
nb_classes = len(categories)

image_w = 100
image_h = 100

pixels = image_h * image_w * 3

X = []
Y = []

for idx, cat in enumerate(categories):
    label = [0 for i in range(nb_classes)]
    label[idx] = 1
    image_dir = caltech_dir + '/' + cat
    files = glob.glob(image_dir + "/*.jpg")
    print(cat, " 파일 길이 : ", len(files))
    for i, f in enumerate(files):
        img = Image.open(f)
        img = img.convert('RGB')
        img = img.resize((image_w, image_h))
        data = np.asarray(img)

        X.append(data)
        Y.append(label)

        if i % 700 == 0:
            print(cat, ':', f)

x = np.array(X)
y = np.array(Y)

print(x.shape) # (200, 100, 100, 3)
print(y.shape) # (200, 4)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)
xy = (x_train, x_test, y_train, y_test)
# np.save('./project/mini/data/multi_image_data.npy', xy)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

print('ok', len(y))