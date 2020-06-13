# 파이썬으로 배우는 딥러닝 교과서

# p417~

# 14.1.1 Pandas로 CSV 읽기
# import pandas as pd

# df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data", header = None)

# # 각 수치가 무엇을 나타내는지 컬럼 헤더로 추가합니다
# df.columns=["", "Alcohol", "Malic acid", "Ash", "Alcalinity of ash", "Magnesium","Total phenols", "Flavanoids", "Nonflavanoid phenols",
# "Proanthocyanins","Color intensity", "Hue", "OD280/OD315 of diluted wines", "Proline"]

# print(df)

import pandas as pd

df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header=None)
df.columns = ["sepal length", "sepal width", "petal length", "petal width", "class"]

print(df)

# p419~
# 14.1.2 CSV 라이브러리로 CSV 만들기
import csv

# with 문을 사용해서 파일을 처리합니다.
with open("csv0.csv", "w") as csvfile:
    # writer() 메서드의 인수로 csvfile과 개행 코드(\n)를 지정합니다
    writer = csv.writer(csvfile, lineterminator='\n')

    # writerow(리스트)로 행을 추가합니다
    writer.writerow(["city", "year", "season"])
    writer.writerow(["Nagano", "1998", "winter"])
    writer.writerow(["Sydney", "2000", "summer"])
    writer.writerow(["Salt Lake City", "2002", "winter"])
    writer.writerow(["Athens", "2004", "summer"])
    writer.writerow(["Torino", "2006", "winter"])
    writer.writerow(["Beiging", "2008", "summer"])
    writer.writerow(["Vancouver", "2010", "winter"])
    writer.writerow(["London", "2012", "summer"])
    writer.writerow(["Sochi", "2014", "winter"])
    writer.writerow(["Rio de Janeiro", "2016", "summer"])

# p421
# 14.1.3 Pandas로 CSV 만들기
data = {'city': ['Nagano', 'Sydney', 'Salt Lake City', 'Athens', 'Torino',
        'Beijing', 'Vancouver', 'London', 'Sochi', 'Rio de Janeiro'],
        'year': [1998, 2000, 2002, 2004, 2006, 2008, 2010, 2012, 2014, 2016],
        'season': ['winter', 'summer', 'winter', 'summer', 'winter', 'summer', 'winter', 'summer', 'winter', 'summer']}
df = pd.DataFrame(data)
df.to_csv('csv1.csv')
print(df)
#              city  year  season
# 0          Nagano  1998  winter
# 1          Sydney  2000  summer
# 2  Salt Lake City  2002  winter
# 3          Athens  2004  summer
# 4          Torino  2006  winter
# 5         Beijing  2008  summer
# 6       Vancouver  2010  winter
# 7          London  2012  summer
# 8           Sochi  2014  winter
# 9  Rio de Janeiro  2016  summer
