import p21_car
import p22_tv

print('=============')
print('do.py의 module 이름은 :', __name__)
print('=============')

p21_car.drive()
p22_tv.watch()


# 같은 폴더내 파일을 불러올 수 있다?

'''
운전하다
car.py의 module 이름은 : p11_car
시청하다
tv.py의 module 이름은 : p12_tv
=============
do.py의 module 이름은 : __main__
=============
운전하다
시청하다

'''