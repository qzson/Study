# 특정 함수 f를 입력하면 f의 결과를 두 배로 만드는 함수를 반환해 주는 고차 함수를 만들고 싶다 해보자.

def doubler(f):
    # f를 참조하는 새로운 함수
    def g(x):
        return 2 * f(x)

        # 새로운 함수를 반환
        return g
# 이 함수는 특별한 경우에만 작동한다.

def f1(x):
    retrun x + 1

g = doubler(f1)
assert g(3) == 8, "(3 + 1) * 2 should equal 8"
assert g(-1) == 0, "(-1 + 1) * 2 should equal 0"

# 두 개 이상의 인자를 받는 함수의 경우에는 문제가 발생한다.
def f2(x, y):
    return x + y

g = doubler(f2)
try:
    g(1,2)
except TypeError:
    print("as defined, g only takes one argument")

# 문제를 해결하기 위해 임의의 수의 인자를 받는 함수를 만들어 줘야 한다.
# 앞서 설명한 언패킹을 사용하면 마법같이 임의의 수의 인자를 받는 함수를 만들 수 있다.
def magic(*args, **kwargs):
    print("unnamed args:", args)
    print("keyword args:", kwargs)

magic(1, 2, key="word", key2="word2")
# unnamed args: (1, 2)
# keyword args: {'key': 'word', 'key2': 'word2'}

def other_way_magic(x, y, z):
    return x + y + z

x_y_list = [1, 2]
z_dict = {"z": 3}
assert other_way_magic(*x_y_list, **z_dict) == 6, "1 + 2 + 3 should be 6"

def doubler_correct(f):
    """f의 인자에 상관없이 작동한다."""
    def g(*args, **kwargs):
        """g의 인자가 무엇이든 간에 f로 보내준다."""
        return 2 * f(*args, **kwargs)
    return g

g = doubler_correct(f2)
assert g(1, 2) == 6, "doubler should work now"