# 20-07-07_30
# vgg16 >> dog, cat 분류


from keras.applications import VGG16
import matplotlib.pyplot as plt

from keras.preprocessing.image import load_img

img_dog = load_img('./data/dog_cat/dog.jpg', target_size=(224,224))
img_cat = load_img('./data/dog_cat/cat.jpg', target_size=(224,224))
img_suit = load_img('./data/dog_cat/suit.jpg', target_size=(224,224))
img_yang = load_img('./data/dog_cat/yang.jpg', target_size=(224,224))

plt.imshow(img_yang)
plt.imshow(img_cat)
# plt.show()


### img 데이터화
from keras.preprocessing.image import img_to_array  # npy 형식으로 변환

arr_dog = img_to_array(img_dog)
arr_cat = img_to_array(img_cat)
arr_suit = img_to_array(img_suit)
arr_yang = img_to_array(img_yang)

print(arr_dog)
print(type(arr_dog))                                # <class 'numpy.ndarray'>
print(arr_dog.shape)                                # (224, 224, 3)

# vgg16 전처리 (그냥 쓰면 문제 있음) // RGB -> BGR
# vgg16에 와꾸대로 잘 넣기 위해서 사용
from keras.applications.vgg16 import preprocess_input

arr_dog = preprocess_input(arr_dog)
arr_cat = preprocess_input(arr_cat)
arr_suit = preprocess_input(arr_suit)
arr_yang = preprocess_input(arr_yang)
# print(arr_dog)                                      # preprocess_input 적용 standardscaler 식의 전처리를 한번 들어간 데이터
# [145. 148. 155.]]] -> [ 51.060997   31.221      21.32     ]]]
# RGB -> GBR 순서가 바뀐 것을 수치상으로 확인 가능
# print(arr_dog.shape)                                # (224, 224, 3)

# img data를 하나로 합친다
import numpy as np
arr_input = np.stack([arr_dog, arr_cat, arr_suit, arr_yang])
                                                    # 현재 data = (1,224,224,3) -> 이미지 4장 = (4,224,224,3)
# print(arr_input.shape)                              # (4, 224, 224, 3) : 전처리가 되어있는 상태


### model 구성
model = VGG16()
probs = model.predict(arr_input)                    # 와꾸 맞춰주었고 가중치는 저장되어 있으니 바로 pred 가능
# print(probs)
print('probs.shape :', probs.shape)                 # probs.shape : (4, 1000)
'''
<1>
[[1.4366491e-07 1.1611222e-08 4.9157461e-07 ... 1.8399254e-09
  2.3695151e-07 1.6579004e-06]

<2>
[1.0318995e-06 4.3254713e-06 1.6692758e-06 ... 8.0409114e-07
  7.9598732e-04 2.6367083e-05]

<3>
[4.2630577e-06 5.6273029e-07 9.6712176e-07 ... 3.3582058e-08
  7.1060441e-07 3.3544213e-05]

<4>
[1.0904158e-06 2.2400548e-06 2.0820648e-06 ... 1.0274141e-06
  3.7983162e-05 1.8630184e-04]]
'''


### img 결과
from keras.applications.vgg16 import decode_predictions

results = decode_predictions(probs)

# list 형태를 하나씩 출력
print('\n-------')
print('dog {}\n'.format(results[0]))
print('cat {}\n'.format(results[1]))
print('suit {}\n'.format(results[2]))
print('yang {}'.format(results[3]))

# >> 배경이 있음에도, 품종별로 잘 분류하고 있다. (5년전 기술)
'''
dog [('n02109961', 'Eskimo_dog', 0.63873285), ('n02110185', 'Siberian_husky', 0.2810013), 
('n02110063', 'malamute', 0.04556877), ('n02091467', 'Norwegian_elkhound', 0.0067409608), 
('n02105412', 'kelpie', 0.005214001)]

cat [('n02123159', 'tiger_cat', 0.34045288), ('n02123045', 'tabby', 0.28112727), 
('n02124075', 'Egyptian_cat', 0.26376718), ('n02127052', 'lynx', 0.04699139), 
('n02114855', 'coyote', 0.0056850337)]

suit [('n02906734', 'broom', 0.33535197), ('n04350905', 'suit', 0.24630642), 
('n03141823', 'crutch', 0.08458366), ('n04367480', 'swab', 0.061475273), ('n03680355', 'Loafer', 0.053831138)]

yang [('n04584207', 'wig', 0.14707425), ('n03000247', 'chain_mail', 0.14465743), 
('n03877472', 'pajama', 0.07643389), ('n03450230', 'gown', 0.06503851), ('n03710637', 'maillot', 0.05737587)]

'''
# 우리 미니 프로젝트에도 적용 가능했을 것
# vgg16을 summary해서 분석 후 사용하면 좋을 것 (그냥 갖다 쓰는 것이 아니라 - 본인에 맞게 커스터마이징 해서)