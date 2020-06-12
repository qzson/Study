# 주제 발표 과제
# 1인 ppt 발표
# 
# 1. 아이디어
# 2. 구현기술
# 3. 계획수립
# 4. 일정
# 5. 최종결과예측


# 20-06-11_23 목요일 // 11:00 ~ 

# 앞으로 더 배워야할 것 
# 비지도 학습 - y값이 없는것
# ??? - AE 오토 인코더도 하고 
# LSTM - 임베딩
#      - 자연어처리..
# CNN - 실제 데이터 처리..
# 우승한 모델들 엮는 것
# tensor 1.14 이후 버전
# 등등

''' xgb 쉐이프 해결 코드? '''
# from sklearn.multioutput import MultiOutputRegressor
# ## 1
# model = MultiOutputRegressor(xgb.XGBRFRegressor())
# model.fit(x_train,y_train)
# score = model.score(x_test,y_test)
# print(score)
# y4 = model.predict(test.values)

# ## 2
# model = MultiOutputRegressor(XGBRFRegressor())
# model.fit(x_train,y_train)
# score = model.score(x_test,y_test)
# print(score)
# y4 = model.predict(test.values)

# 20-06-11_23 PCA를 통한 차원 축소 > 다시 복원
# y 타겟값만 PCA를 한다
# x_pred 로 나온 y_pred를 다시 pca해서 원복한다

