# 20-07-13_34
# route 2

from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello333():
    return '<h1>hello world</h1>'

@app.route('/ping', methods=['GET'])            # GET 방식으로 땡겨 오겠다. post도 있기는 함
def ping():
    return '<h1>pong</h1>'

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False)
