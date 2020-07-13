# 20-07-13_34
# response2

from flask import Flask, Response, make_response
app = Flask(__name__)



@app.before_first_request
def before_first_request():
    print('[1] 앱이 가동되고 나서 첫번째 HTTP 요청에만 응답합니다.')

@app.before_request
def before_request():
    print('[2] 매 HTTP 요청이 처리되기 전에 실행됩니다.')

@app.route('/')
def response_test():
    custom_response = Response('Custom Response', 200, {'Program' : 'Flask Web Application'})
    print('★')
    return make_response(custom_response)

@app.after_request
def after_request(response):
    print('[3] 매 HTTP 요청이 처리되고 나서 실행됩니다.')
    return response

@app.teardown_request
def teardown_request(exception):
    print('[4] 매 HTTP 요청의 결과가 브라우저에 응답하고 나서 호출된다.')

@app.teardown_appcontext
def teardown_appcontext(exception):
    print('[5] HTTP 요청의 애플리케이션 컨텍스트가 종료될 때 실행된다.')

if __name__ == '__main__':
    app.run(host='127.0.0.1')   # flask의 기본 port는 5000

# app.route 작업이 되기전에 before_first_request가 구동된다.
# 그래서 print문이 먼저 생성되고 웹이 열린다

'''
과제 tf rnn구현
쿠키, session 정리
reponse, request 정리
'''