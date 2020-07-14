# 20-07-14_35
# database 파일을 넘파이로 저장

import pymssql as ms
import numpy as np

conn = ms.connect(server='127.0.0.1', user='bit2', password='3411', database='bitdb')

cursor = conn.cursor()

cursor.execute('SELECT * FROM iris2;')

row = cursor.fetchall()
print(row)
conn.close()

print('===NUMPY===\n')
dataset = np.asarray(row)
print(dataset)
print(dataset.shape)
print(type(dataset))

np.save('./data/test_flask_iris2.npy', dataset)
print('DATA SAVE COMPLITE')

# 각 테이블을 만들고 각각 np로 변환해서 작업
