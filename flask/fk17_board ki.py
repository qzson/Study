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

# 데이터베이스 만들기
conn = sqlite3.connect("./data/wanggun.db")
cursor = conn.cursor()
cursor.execute("SELECT * FROM general;")
print(cursor.fetchall())

@app.route('/')
def run():
    conn = sqlite3.connect('./data/wanggun.db')
    c = conn.cursor()
    c.execute("SELECT * FROM general;")
    rows = c.fetchall()
    return render_template("board_index_ki.html", rows=rows)

@app.route('/modi')
def modi():
    ids = request.args.get('id')
    conn = sqlite3.connect('./data/wanggun.db')
    c = conn.cursor()
    c.execute('SELECT * FROM general where id = ' + str(ids))
    rows = c.fetchall()
    return render_template('board_modi_ki.html', rows=rows)

@app.route('/addrec', methods=['POST', 'GET'])
def addrec():
    if request.method == 'POST':
        try:
            conn = sqlite3.connect('./data/wanggun.db')
            war = request.form['war']
            ids = request.form['id']
            print('11')
            c = conn.cursor()
            c.execute('UPDATE general SET war = '+ str(war) + " WHERE id = "+str(ids))
            conn.commit()
            msg = '정상적으로 입력되었습니다.'
        except:
            conn.rollback()
            msg = '에러가 발생하였습니다.'
        finally:
            conn.close()
            return render_template("board_result_ki.html", msg=msg)

app.run(host='127.0.0.1', port=5000, debug=False)