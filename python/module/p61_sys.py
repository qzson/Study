import sys
print(sys.path)

from test_import import p62_import
p62_import.sum2()
# 이 import는 아나콘다 폴더 들어있다
# 작업그룹 임포트 썸탄다.

# '파일'을 임포트 했기 때문에 2개의 print 값이 출력


print('=-=-=-=-=-=-=-=')

from test_import.p62_import import sum2
sum2()
# 작업그룹 임포트 썸탄다.

# '함수'를 임포트 했기 때문에 1개의 print 값이 출력