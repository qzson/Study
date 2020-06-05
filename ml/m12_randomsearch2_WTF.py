# 20-06-05 // 오후

# RandomizedSearch 란

# 그리드서치의 일부가 RandomizedSearch

'''

Very similar to grid search:
Define an estimator, which hyperparameters to tune and the range of values for each hyperparameter
We still set a cross-validation scheme and scoring function. BUT, we instead randomly select grid squares

Why does this work ?
- Randomly chosen trials are more effcient for hyper-parameter-optimization
Two main reasons:
- 1. Not every hyperparameter is as important
- 2. A little trick of probability

'''

# 그리드서치의 일부를 가져다 쓰는 것이 랜덤아이즈드서치
# 케라스에서는 너무 연산이 많기 때문에 이걸 쓴다
# 랜덤아이즈드서치와 그리드서치 간 성능차이가 크게 나지 않는다.

# 다음주에는 이것을 엮어서 데이터 전처리를 엮는 파이프라인과 케라스와 엮는다.
# + Feature importance