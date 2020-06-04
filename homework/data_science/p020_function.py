# p20, 2.6 함수
# 함수란 0개 혹은 그 이상의 인자를 입력 받아 결과를 반환하는 규칙이다.
# 파이썬에서는 def를 이용해 함수를 정의한다.


def double(x):
    """
    이곳은 함수에 대한 설명을 적어 놓는 공간이다.
    예를 들어 '이 함수는 입력된 변수에 2를 곱한 값을 출력해 준다.'라는 설명을 추가할 수 있다.
    """
    return x * 2


def apply_to_one(f):
    """인자가 1인 함수 f를 호출"""
    return f(1)


my_double = double           # 방금 정의한 함수 나타냄
x = apply_to_one(my_double)
print(x)                     # 2

# lambda function (람다 함수)

y = apply_to_one(lambda x: x + 4)  # 5
print(y)

# another_double = lambda x: 2 * x)    # 이 방법은 최대한 피하도록
def another_double(x):
    """ 대신 이렇게 작성하자. """
    return 2 * x
print(x)                            # 2


def my_print(message="my default message"):
    print(message)

my_print("hello")         # hello
my_print()                # my default message

def full_name(first = "what's-his-name", last = "Something"):
    return first + " " + last

full_name("Joel", "Grus") # 'Joel Grus'를 출력
full_name("Joel")         # 'Joel Something'을 출력
full_name(last="Grus")    # 'What;s-his-name Grus'를 출력