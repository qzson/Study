# 20-06-08_20
# 즐거운 월요일 시작
# 저번주 리뷰

''' 저번주 리뷰

- 인공지능의 암흑기 : xor에서 부터 선을 그리기 힘들어졌다 -> 해결
- 각 모델을 머신러닝 모델을 5~6개 정도를 케라스로 변환해봤다. 모델 자체는 디폴트로 구성했지만, 간단했고 속도는 매우 빨랐다
- 머신러닝은 cpu를 사용한다 (n_jobs (시피유 연산속도 증가) : 랜덤포레스트에서 사용했었다)
- y 데이터(타겟데이터) 작업 함에 있어서 y 값의 범위 조정 (타겟값 조정으로 acc를 높혔다)
    >> 잘못 바꾸면 오더의 의도를 바꿀 가능성이 있다

- all_estimators로 여러가지 머신러닝 모델에 대한 score 값을 비교했었다
- 회귀모델인지 분류모델인지 확인할 수 있다 (로지스틱리그레서는 분류인점)
- 회귀모델 score (R2) / 분류모델 score (acc) //
    >> acc가 1.0인데 값이 이상한 경우 등의 오류에 빠지면 안되며 반드시 비교를 해봐야한다

- 머신러닝 모델에서는 해도 되긴 하지만, 원핫인코딩에 대해서 자유로웠다 (y값 라벨링에 대해서) [질문필요]
- 딥러닝의 데이터, 모델구성, 훈련, 평가예측 >> 머신러닝에서도 비슷하게 적용 구성했다
    >> 데이터, 모델구성, 파라미터, 모델fit 동일했고, 스코어(evaluate와 동일)
    >> model.fit -> fit // evaluate -> score // predict -> predict // (케라스와 머신러닝과 문법이 유사하다)
    >> score 부분만 조심 // 케라스 evaluate 의 반환값 'loss','metrics' 리스트로 던져준다.
    >> fit의 반환값 'history' -> checkpoint 했었고

- kfold / cv=5 전체 5조각으로 작업 진행

- gridsearch / randomsearch
    >> gridsearch : 전체 다한다 // randomsearch : 일부만 빼와서 한다
    >> 성능은 비슷하지만 random 형식이 빠르다
    >> randomforest에 대해서 조금 친숙해진 시간이었다

- 우리가 전처리 시, pred하는 y값은 주어지지 않는다
    >> train, test, (조정이 가능한 부분) // pred_x (대회에서 요구되는 결과값)
    >> 데이터 전처리 : minmax, standard, pca... 등이 있었지만, (scale.fit : train만 / scale.transform : train, test, pred, ... [y(타겟값)은 건들지 않는다])
    >> y값은 분류에서의 원핫인코딩 할 때 빼고는 건들지 않는다

- 이상치, 결측치 부분에서 본의 아니게 만들어진 결측치가 있었다
    >> 엄밀히 말해서는 데이터라 제거해서는 안되지만, 결측치는 수정하는 방법이 있었다
    >> 1. 제거 (x, y 둘다 없는 것은 결측치가 아니라 아예 데이터가 없는 것)
    >> 중간 중간 x는 있는데 y가 NaN 인 경우...
    >> 1. 제거 / 2. 0 기입 / 3. 평균 / 4. 위, 아래 값 넣기 5. NaN 외 부분을 train을 넣고 빈 부분을 test와 pred로 예측 // 정도 했었다
        >> pandas에서 제공하는 기본적인 것을 사용하면 평타 85% 넘는다

- 오늘 머신러닝에서 들어갈 부분은
    >> grids, randoms 를 케라스와 엮고 parameter를 거의 막 집어넣고 돌릴 것
    >> 머신러닝의 파이프라인이라는 기능
        >> kfold
        >> 전처리 minmax, standard 이것도 우리가 했던 grids 에 포함된다
    >> PCA에 대한 또다른 모델
        PCA와 쌍벽을 이루는 디시져 서트리?, 랜덤포레스트, XG부스터의 'Feature importance'
    >> XG부스터는 병렬식 코어를 진행 따라서, 우리도 n_jobs가 적용된다

- 다시 딥러닝 들어가서 우리가 못했던 부분 가중치에 대해서
    >> 가중치에 대한 계산, activation이나 loss에 대한 계산
    >> 이미지를 책에 있는 예제가 아닌 실제 이미지로 제네레이터 써서 수치로 바꾼후 작업을 할 것이다
        > 수치 이미지 쓰는 것도 conv2D 사용했었는데 conv1D 사용할 것이고 수치가 lstm?에서 어떻게 변하는지
    >> lstm?과 인베디드 까지 tensorflow 1.0

'''