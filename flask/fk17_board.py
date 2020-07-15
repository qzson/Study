# 20-07-15_36
# board web 상에서 데이터 수정해보기

'''
GET  : 모든 파라미터를 url로 보내는 것 (눈에 보임)
POST : 전달하려는 정보가 HTTP body에 포함되어 전달되는 것 (눈에 보이지 않음)
https://amudabadmus.files.wordpress.com/2017/03/web-crawling-scraping-ajax-sites-3-638.jpg?w=638
'''

from flask import Flask, render_template, request
import sqlite3

app = Flask(__name__)

# database browser 
conn = sqlite3.connect("./data/wanggun.db")                     # data 폴더에 해당 data파일 넣기
cursor = conn.cursor()
cursor.execute("SELECT * FROM general;")                        # general 에 있는 모든 데이터를 가지고 오게 된다. 
print(cursor.fetchall())

@app.route('/')                                                 # local host 주소만 치면 결과 출력 
def run():
    conn = sqlite3.connect('./data/wanggun.db')
    c = conn.cursor()
    c.execute("SELECT * FROM general;")
    rows = c.fetchall()
    return render_template("board_index_ki.html", rows = rows)  # html파일 생성, 해당 html은 rows를 받아준다. 

# /modi
@app.route('/modi')
def modi():
    ids = request.args.get('id')
    conn = sqlite3.connect('./data/wanggun.db')
    c = conn.cursor()
    c.execute('SELECT * FROM general where id = ' + str(ids))   # id 를 설정해주면 설정한 id에 대한 값만 출력. 
    rows = c.fetchall()
    return render_template('board_modi_ki.html', rows = rows)

# /addrec
@app.route('/addrec', methods=['POST', 'GET'])
def addrec():
    if request.method == 'POST':
        try:
            war = request.form['war']
            ids = request.form['id']
            with sqlite3.connect('./data/wanggun.db') as conn:
                cur = conn.cursor()
                cur.execute('UPDATE general SET war = ' + str(war) + ' WHERE id = ' + str(ids))
                conn.commit()
                msg = '정상적으로 입력되었습니다.'
        except:
            conn.rollback()
            msg = '입력과정에서 에러가 발생했습니다.'
        
        finally:
            return render_template('board_result_ki.html', msg = msg)
            conn.close()

app.run(host='127.0.0.1', port = 5000, debug = False)