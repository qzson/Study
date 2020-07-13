# 20-07-13_34
# 현재 작업 폴더 하단에 있으면 작동이 된다.
# 플라스크 폴더 하단에 template라는 폴더를 만들어준다.

from flask import Flask, render_template
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/user/<name>')
def user(name):
    return render_template('user.html', name=name)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False)
