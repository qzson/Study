'''
데이터를 시각화하기 위한 도구는 무궁무진하다.
matplotlib은 웹을 위한 복잡하고 인터렉티브한 시각화를 만들고 싶다면 가장 좋은 선택은 아닐 수 있다.
하지만, 간단한 막대, 선 그래프 or 산점도를 그릴 때는 나쁘지 않다.
'''

''' 선 그래프 만들기 '''

from matplotlib import pyplot as plt

years = [1950, 1960, 1970, 1980, 1990, 2000, 2010]
gdp = [300.2, 543.3, 1075.9, 2862.5, 5979.6, 10289.7, 14958.3]

# x축에 연도, y축에 GDP가 있는 선 그래프를 만들자.
plt.plot(years, gdp, color='green', marker='o', linestyle='solid')

# 제목을 더하자.
plt.title("Nominal GDP")

# y축에 레이블을 추가하자.
plt.ylabel("Billions of $")
plt.show()


''' 막대 그래프 만들기 '''

movies = ["Annie Hall", "Ben-Hur", "Casablanca", "Gandhi", "West Side Story"]
num_oscars = [5, 11, 3, 8, 10]

# 막대의 x 좌표는 [0, 1, 2, 3, 4], y 좌표는 [num_oscars]로 설정
plt.bar(range(len(movies)), num_oscars)

plt.title("My Favorite Movies")     # 제목을 추가
plt.ylabel("# of Academy Awards")   # y축에 레이블을 추가하자

# x축 각 막대의 중앙에 영화 제목을 레이블로 추가하자
plt.xticks(range(len(movies)), movies)

plt.show()