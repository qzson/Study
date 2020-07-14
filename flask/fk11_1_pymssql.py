# 20-07-14_35
# ssms 데이터 가져오기
# pip install pymssql


import pymssql as ms

conn = ms.connect(server='127.0.0.1', user='bit2', password='3411', database='bitdb')   # mssql에 연결
# server=localhost 도 가능

cursor = conn.cursor()

cursor.execute('SELECT * FROM iris2;')

# 150행 중 1줄을 가져온다 # .fetchone() : 한줄을 가져 온다
row = cursor.fetchone()
# row = cursor.fetchchone()
# row = cursor.fetchchone()

while row :
    print('첫 c : %s, 둘 c : %s 셋 c : %s'%(row[0], row[1], row[2]))
    row = cursor.fetchone()

conn.close()    # connect 했으니 close
