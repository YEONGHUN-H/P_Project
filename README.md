[BMI DATA 확인]

1. 데이터 탐색 및 준비
* 데이터 확인: 741명의  키, 몸무게, BMI, 성별, 나이 등의 정보가 있음
* 결측치 처리: 데이터에서 누락된 값이 있는지 확인하고, 필요하면 채우거나 제거
* 데이터 변환: BMI 값에 따라 사용자들을 분류하거나, 나이, 성별, 체중 등을 바탕으로 새로운 피처를 생성할 수 있음.
* 관련 코드 부분 예시:https://www.kaggle.com/code/ahthneeuhl/bmi-data-analysis-visualization-and-predictions
* 데이터셋 500명: https://www.kaggle.com/datasets/yersever/500-person-gender-height-weight-bodymassindex

2. 운동 추천 기준 설정
* 운동 추천 로직: BMI와 같은 데이터를 기반으로 다음과 같은 운동을 추천할 수 있음.
    * 저체중: 체중 증가를 위한 근력 운동 + 고칼로리 식단.
    * 정상체중: 균형 잡힌 운동(유산소 + 근력 운동).
    * 과체중: 체중 감소를 위한 유산소 운동 중심 프로그램.
    * 비만: 저강도 유산소 + 식단 조절.
* 기타 기준: 성별, 나이, 활동 수준에 따라 추천을 세분화합니다.

2-1. 식단 데이터 추가하기
만약 신단 데이터 받는다면 , 자료 조사를 하고 아침 점심 저녁 저체중 기준 식단 칼로리 단백질 구성에 맞춰서 진행해서 추천해주는 기능 추가할거임.  (식품명,에너지,지방,당류,탄수화물) 등 정보 바탕으로 추천하게 구성해야함.

2-2. 칵테고리별 나눠서 목표를 주고 그거로 하기.
목표를 근성장,다이어트,유지 
근성장: 단백질 , 탄수화물
다이어트: 영양소 있으면서 기초대사량보다 적은 양의 다이어트 (기초대사량 노드 보면 어캐 구하는지 있음)
-> 기초대사량 구하는 식 있으니까 그거 계산해서 항목에 추가해서 진행. ( 기초 대사량보다 많이 먹기)
아침 추천 앱으로 진행하는데 (10개 이렇게 아침으로 뽑아두고 랜덤으로 추천되게)
-> 데이터 가공하는 과정에서 아침 점심 저녁 으로 나누기.
-> 데이터 100그람당 칼로리라서 1인분 칼로리 따로 항목 만들면 좋을듯.

회의록에 작성 (뭐했는지)
=========================================================================
<데이터 가공 정리>
1. Bmi 데이터를 이용하여 과체중,정상,저체중,비만인지 알려줌. (BMI 계산법 적용해서 보여줄 수 있게.) 
2. 다이어트,유지,벌크업 3가지 항목을 만들어서 각 항목당 BMI Class에 맞게 운동과 식단을 추천해줄거임.
3. 식단은 (영양소)를 기준으로 해서 추천하도록 구성해야함.
4. 운동은 근력,유산소 두가지 항목으로 구성함.

지금 식단이 너무 한끼에 1키로니까 이거 잘 수정해서 (실제 앱 식단 추천 앱 보고 한번 따라해보기)

키토 식단 (전문가 식단으로 진짜 샐러드 이렇게 추천해주는 형태로 진행 식단 전처리에 전문가로 추가하고
다이어트 유지 유형에 전문가 입력하면 키토 식단으로 추가해주는거)
<운동 데이터>
상 중 하 나눠놓고 칵테고리 나눠져 있으니까 그걸 사용하면 학습 시킬 수 있는거지.


=========================================================================
3. 모델 개발
* 데이터 라벨링: 기존 데이터를 이용해 운동 추천을 위한 라벨을 추가. 예를 들어, "운동 유형" 컬럼에 "유산소", "근력", "혼합" 등의 값을 할당. 식단은 “채식”, “육식”
* 모델 학습:
    * 머신러닝 알고리즘(예: 분류 모델): 사용자 데이터를 입력하면 운동 유형을 예측하도록 학습.
    * 추천 시스템: 비슷한 BMI, 나이, 성별을 가진 사용자 그룹을 생성하고 운동과 식단을 추천.

4. 예측 및 평가
* 모델 테스트: 새로운 사용자 데이터를 입력해 운동 추천 결과를 확인할 예정.
* 모델 평가: 정확도, F1-score 등을 통해 추천 정확도를 평가할 예정.