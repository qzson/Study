# 20-07-14_35
# sqlite3
'''
# DB Browser for SQLite
1. 새 데이터 베이스
2. 데이터 보기
'''
import sqlite3

conn = sqlite3.connect('test.db')    # 만든적 없으면 자동 생성

cursor = conn.cursor()

# supermarket TABLE안에 Itemno...Price가 들어 있다
cursor.execute('''CREATE TABLE IF NOT EXISTS supermarket(Itemno INTEGER, Category TEXT, FoodName TEXT, Company TEXT, Price INTEGER)''')

# 한 번 더 실행시키면 데이터가 겹치게 된다 (기존에 있던 것을 지울 것 ???)
sql = 'DELETE FROM supermarket'
cursor.execute(sql)

# 데이터 넣기
sql = 'INSERT into supermarket(Itemno, Category, FoodName, Company, Price) values (?,?,?,?,?)'
cursor.execute(sql, (1, '과일', '자몽', '마트', 1500))

sql = 'INSERT into supermarket(Itemno, Category, FoodName, Company, Price) values (?,?,?,?,?)'
cursor.execute(sql, (2, '음료수', '망고주스', '편의점', 1000))

sql = 'INSERT into supermarket(Itemno, Category, FoodName, Company, Price) values (?,?,?,?,?)'
cursor.execute(sql, (33, '고기', '소고기', '하나로마트', 10000))

sql = 'INSERT into supermarket(Itemno, Category, FoodName, Company, Price) values (?,?,?,?,?)'
cursor.execute(sql, (4, '약', '박카스', '약국', 500))

sql = 'SELECT * FROM supermarket'
# sql = 'SELECT Itemno, Category, FoodName, Company, Price FROM supermarket'

cursor.execute(sql)

rows = cursor.fetchall()

for row in rows:
    print(str(row[0]) + ' ' + str(row[1]) + ' ' + str(row[2]) + ' ' + str(row[3]) + ' ' + str(row[4]))

conn.commit()   # DB browser sql에 보낸다
conn.close()