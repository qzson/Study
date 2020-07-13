# 20-07-13_34

from flask import Flask
app = Flask(__name__)

from flask import make_response

@app.route('/')
def index():
    response = make_response('<h1> 잘 따라 치시오!!! </h1>')
    response.set_cookie('answer', '42')
    return response

if __name__=='__main__':
    app.run(host='127.0.0.1', port=5000, debug=False)

'''
Cookie
쿠키는 클라이언트의 PC에 텍스트 파일 형태로 저장되는 것으로 일반적으로는 시간이 지나면 소멸한다.
보통 세션과 더불어 자동 로그인, 팝업 창에서 "오늘은 이 창을 더 이상 보지 않기" 등의 기능을 클라이언트에 저장해놓기 위해 사용된다.

웹 페이지에서 폼을 전송받으며, 클라이언트에 쿠키를 넘겨주는 코드를 짜보자. 


Flask에서 set_cookie로 쿠키를 생성하고, request.cookies.get() 를 통해 쿠키를 불러 올 수 있습니다.

<set_cookie(key, value='', max_age=None, expires=None, path='/', domain=None, secure=None, httponly=False)>

* Key = 설정되는 쿠키의 키 (이름)
* Value = 쿠키의 값
* Max_age = 초 단위, 쿠키가 클라이언트의 브라우저 세션만큼 지속되는 값
* expires = datetime 객체 / UNIX Timestamp
* domain = 도메인간 쿠키 설정시
* path = 쿠키를 지정된 경로로 제한

'''