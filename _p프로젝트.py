# -*- coding: utf-8 -*-
"""P프로젝트.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1vruVBDixLwhETcemLrrg_m2n98wMo3qU
"""

import pandas as pd
#####데이터 가공하는 첫번째 과정 ######
input_file = '/content/drive/MyDrive/Colab Notebooks/P프로젝트/bmi.csv'
output_file = '/content/drive/MyDrive/Colab Notebooks/P프로젝트/gender.csv'
df = pd.read_csv(input_file)

# Gender 열 추가 (Height 기준으로 1.70m 밑은 Female, 이상은 Male)
df['Gender'] = df['Height'].apply(lambda x: 'Female' if x < 1.70 else 'Male')

print(df.head(5))

df.to_csv(output_file, index=False)
print(f"가공 완료. 데이터가 '{output_file}'에 저장되었습니다.")

"""#1. 라이브러리 및 데이터 경로 설정"""

# 필요한 라이브러리 불러오기
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# 데이터 파일 경로
bmi_data_path = '/content/drive/MyDrive/Colab Notebooks/P프로젝트/gender.csv'
food_data_path = '/content/drive/MyDrive/Colab Notebooks/P프로젝트/final_food_data.csv'

# 데이터 읽기
bmi_data = pd.read_csv(bmi_data_path)
food_data = pd.read_csv(food_data_path)

# 데이터 확인
print("BMI 데이터 예시:")
print(bmi_data.head())
print("\n식품 데이터 예시:")
print(food_data.head())

"""# 2.BMI 데이터 전처리

"""

# BMR 계산 함수 (Harris-Benedict 공식)
def calculate_bmr(weight, height, age, gender):
    if gender == 'Male':
        return 88.362 + (13.397 * weight) + (4.799 * height) - (5.677 * age)
    else:
        return 447.593 + (9.247 * weight) + (3.098 * height) - (4.330 * age)

# BMI 상태와 목표에 따른 칼로리 목표 설정 함수
def get_calorie_target(bmi_class, bmr, goal):
    if goal == '다이어트':
        return bmr - 500 if bmi_class != '저체중' else bmr - 300
    elif goal == '유지':
        return bmr
    elif goal == '벌크업':
        return bmr + 500 if bmi_class != '비만' else bmr + 300

# BMR 및 칼로리 목표 계산
bmi_data['BMR'] = bmi_data.apply(lambda row: calculate_bmr(row['Weight'], row['Height'] * 100, row['Age'], row['Gender']), axis=1)
bmi_data['Calorie_Target'] = bmi_data.apply(lambda row: get_calorie_target(row['BmiClass'], row['BMR'], '다이어트'), axis=1)

print("BMR 및 칼로리 목표 추가된 BMI 데이터:")
print(bmi_data.head())

"""#2-1. 식품 데이터 전처리"""

import pandas as pd

# 데이터 로드
food_data_path = '/content/drive/MyDrive/Colab Notebooks/P프로젝트/final_food_data.csv'  # 식품 데이터 경로
food_data = pd.read_csv(food_data_path, encoding='utf-8')

# 식품 데이터 전처리
food_data['식품중량'] = food_data['식품중량'].str.replace('ml', 'g').str.replace('m', '').str.replace('g', '')
food_data['식품중량'] = pd.to_numeric(food_data['식품중량'], errors='coerce')
food_data = food_data.dropna(subset=['식품중량'])

# 100g 기준으로 영양소 계산
food_data['칼로리_100g'] = (food_data['에너지(kcal)'] / food_data['식품중량']) * 100
food_data['탄수화물_100g'] = (food_data['탄수화물(g)'] / food_data['식품중량']) * 100
food_data['단백질_100g'] = (food_data['단백질(g)'] / food_data['식품중량']) * 100
food_data['지방_100g'] = (food_data['지방(g)'] / food_data['식품중량']) * 100

# 이상치 제거 (칼로리 800 이상 제외)
filtered_food_data = food_data[food_data['칼로리_100g'] < 800]

# 음식 분류 함수
def classify_food(row):
    if any(x in row['식품대분류명'] for x in ["밥류", "면 및 만두류"]):
        return "밥류"
    elif any(x in row['식품대분류명'] for x in ["국 및 탕류", "찌개 및 전골류"]):
        return "국류"
    elif any(x in row['식품대분류명'] for x in [
        "전·적 및 부침류", "조림류", "나물·숙채류", "튀김류", "구이류",
        "장류", "양념류", "찜류", "볶음류", "생채·무침류",
        "젓갈류", "김치류", "장아찌·절임류"]):
        return "반찬류"
    elif any(x in row['식품대분류명'] for x in [
        "빵 및 과자류", "음료 및 차류", "유제품류 및 빙과류", "샌드위치", "곡류, 서류 제품"]):
        return "디저트류"
    elif any(x in row['식품대분류명'] for x in ["브런치", "샌드위치"]):
        return "브런치류"
    else:
        return "기타"

# 음식 분류 열 추가
filtered_food_data['음식분류'] = filtered_food_data.apply(classify_food, axis=1)

# 결과 확인
print("음식 분류 결과:")
print(filtered_food_data['음식분류'].value_counts())

"""#머신러닝 모델 학습

#3. 데이터 준비
"""

# 학습 데이터 생성
X = filtered_food_data[['칼로리_100g', '탄수화물_100g', '단백질_100g', '지방_100g']]
y = filtered_food_data['식품명']

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"""#3-1. 모델 학습

"""

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import joblib

# 데이터 전처리: 정규화
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 하이퍼파라미터 튜닝: 교차 검증을 통한 최적의 n_neighbors 찾기
param_grid = {
    'n_neighbors': range(1, 21),  # 1부터 20까지의 n_neighbors를 탐색
    'weights': ['uniform', 'distance'],  # 균등 가중치와 거리 기반 가중치 비교
    'metric': ['euclidean', 'manhattan']  # 거리 계산 방식 비교
}
knn = KNeighborsClassifier()
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)

# 최적의 하이퍼파라미터 출력
best_params = grid_search.best_params_
print(f"최적의 하이퍼파라미터: {best_params}")

# 최적의 모델로 학습
best_knn = grid_search.best_estimator_
print("최적 모델 학습 완료!")

# 모델 평가
y_pred = best_knn.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"테스트 세트 정확도: {accuracy:.2f}")
print("분류 보고서:")
print(classification_report(y_test, y_pred))

# 모델 저장
joblib.dump(best_knn, 'best_knn_model.pkl')
print("모델이 저장되었습니다!")

"""#추천 시스템 구축

# 4.사용자 입력 및 칼로리 계산
"""

# BMI 계산 함수
def calculate_bmi(weight, height):
    bmi = weight / ((height / 100) ** 2)  # 키를 cm에서 m로 변환하여 계산
    if bmi < 18.5:
        return '저체중', bmi
    elif 18.5 <= bmi < 23:
        return '정상체중', bmi
    elif 23 <= bmi < 25:
        return '과체중', bmi
    else:
        return '비만', bmi

# BMR 계산 함수 (Harris-Benedict 공식)
def calculate_bmr(weight, height, age, gender):
    if gender == 'Male':
        return 88.362 + (13.397 * weight) + (4.799 * height) - (5.677 * age)
    elif gender == 'Female':
        return 447.593 + (9.247 * weight) + (3.098 * height) - (4.330 * age)
    else:
        raise ValueError("성별은 'Male' 또는 'Female'로 입력해야 합니다.")

# TDEE 계산 함수
def calculate_tdee(bmr, activity_level):
    activity_level_mapping = {1: 1.2, 2: 1.375, 3: 1.55, 4: 1.725}
    if activity_level not in activity_level_mapping:
        raise ValueError("활동 수준은 1~4 사이의 정수여야 합니다.")
    return bmr * activity_level_mapping[activity_level]

# 목표별 영양소 비율 설정
goal_ratios = {
    "다이어트": {"carb_ratio": 0.5, "protein_ratio": 0.3, "fat_ratio": 0.2},
    "유지": {"carb_ratio": 0.55, "protein_ratio": 0.25, "fat_ratio": 0.2},
    "벌크업": {"carb_ratio": 0.6, "protein_ratio": 0.3, "fat_ratio": 0.1}
}

# 사용자 입력
user_info = {
    'weight': 70,
    'target_weight': 75,  # 목표 체중
    'height': 175,
    'age': 30,
    'gender': 'Male',
    'goal': '유지',  # 다이어트, 유지, 벌크업
    'activity_level': 3
}

# 현재 체중 BMI, BMR 및 TDEE 계산
current_bmi_class, current_bmi = calculate_bmi(user_info['weight'], user_info['height'])
current_bmr = calculate_bmr(user_info['weight'], user_info['height'], user_info['age'], user_info['gender'])
current_tdee = calculate_tdee(current_bmr, user_info['activity_level'])

# 목표 체중 BMI, BMR 및 TDEE 계산
target_bmi_class, target_bmi = calculate_bmi(user_info['target_weight'], user_info['height'])
target_bmr = calculate_bmr(user_info['target_weight'], user_info['height'], user_info['age'], user_info['gender'])
target_tdee = calculate_tdee(target_bmr, user_info['activity_level'])

# 목표 설정에 따른 영양소 비율 적용
ratios = goal_ratios[user_info['goal']]

# 현재 체중 기반 영양소 목표 계산
current_carb_target = (current_tdee * ratios['carb_ratio']) / 4
current_protein_target = (current_tdee * ratios['protein_ratio']) / 4
current_fat_target = (current_tdee * ratios['fat_ratio']) / 9

# 목표 체중 기반 영양소 목표 계산
target_carb_target = (target_tdee * ratios['carb_ratio']) / 4
target_protein_target = (target_tdee * ratios['protein_ratio']) / 4
target_fat_target = (target_tdee * ratios['fat_ratio']) / 9

# 결과 출력
print("=== 현재 체중 정보 ===")
print(f"현재 BMI: {current_bmi:.2f} ({current_bmi_class})")
print(f"현재 TDEE: {current_tdee:.2f} kcal")
print(f"현재 탄수화물 목표: {current_carb_target:.2f} g")
print(f"현재 단백질 목표: {current_protein_target:.2f} g")
print(f"현재 지방 목표: {current_fat_target:.2f} g")

print("\n=== 목표 체중 정보 ===")
print(f"목표 BMI: {target_bmi:.2f} ({target_bmi_class})")
print(f"목표 TDEE: {target_tdee:.2f} kcal")
print(f"목표 탄수화물 목표: {target_carb_target:.2f} g")
print(f"목표 단백질 목표: {target_protein_target:.2f} g")
print(f"목표 지방 목표: {target_fat_target:.2f} g")

"""#4-1. 메뉴 추천 함수"""

import random

# 끼니별 비율 설정
meal_ratios = {
    "breakfast": 0.3,  # 아침 30%
    "lunch": 0.4,      # 점심 40%
    "dinner": 0.3      # 저녁 30%
}

# 간식 추천 비율 설정
snack_ratios = {
    "다이어트": {"morning_snack": 0.05, "afternoon_snack": 0.05, "evening_snack": 0.0},
    "유지": {"morning_snack": 0.1, "afternoon_snack": 0.1, "evening_snack": 0.05},
    "벌크업": {"morning_snack": 0.1, "afternoon_snack": 0.1, "evening_snack": 0.1},
}

# 식사 추천 함수
def recommend_meal(calorie_target, food_data, used_foods, carb_ratio, protein_ratio, fat_ratio):
    """
    끼니별 식사 추천: 밥류, 국류, 반찬류에서 랜덤으로 선택.
    각 항목은 사용된 음식 목록에 추가하며, 적절한 섭취량을 계산.
    """
    recommendations = []
    portion_limits = {"밥류": 300, "국류": 250, "반찬류": 200}
    for category in ["밥류", "국류", "반찬류"]:
        filtered_data = food_data[(food_data['음식분류'] == category) & (~food_data['식품명'].isin(used_foods))]
        if not filtered_data.empty:
            food = filtered_data.sample(1).iloc[0]
            portion = min(calorie_target / food['칼로리_100g'] * 100, portion_limits[category])
            recommendations.append(f"{food['식품명']} {portion:.1f}g")
            used_foods.append(food['식품명'])
    return recommendations

# 간식 추천 함수
def recommend_snack(calorie_target, food_data, used_foods, max_portion=100):
    """
    간식 추천: 디저트류에서 랜덤으로 선택하며, 섭취량 계산.
    섭취량이 max_portion을 초과하면 '마음껏'으로 출력.
    """
    snack_data = food_data[food_data['음식분류'] == '디저트류']
    valid_snacks = snack_data[~snack_data['식품명'].isin(used_foods)]

    if valid_snacks.empty:
        return "추천 가능한 간식 없음"

    snack = valid_snacks.sample(1).iloc[0]
    portion = calorie_target / snack['칼로리_100g'] * 100
    used_foods.append(snack['식품명'])

    if portion > max_portion:
        return f"{snack['식품명']} 마음껏"
    else:
        return f"{snack['식품명']} {portion:.1f}g"

# 사용된 음식 기록 및 칼로리 계산
used_foods = []
used_calories = 0

# 사용자 목표에 따른 끼니와 간식 추천
breakfast = recommend_meal(user_calorie_target * meal_ratios["breakfast"], filtered_food_data, used_foods,
                           carb_ratio, protein_ratio, fat_ratio)
used_calories += user_calorie_target * meal_ratios["breakfast"]

if snack_ratios[user_goal]["morning_snack"] > 0:
    morning_snack = recommend_snack(user_calorie_target * snack_ratios[user_goal]["morning_snack"],
                                    filtered_food_data, used_foods)
    used_calories += user_calorie_target * snack_ratios[user_goal]["morning_snack"]

lunch = recommend_meal(user_calorie_target * meal_ratios["lunch"], filtered_food_data, used_foods,
                       carb_ratio, protein_ratio, fat_ratio)
used_calories += user_calorie_target * meal_ratios["lunch"]

if snack_ratios[user_goal]["afternoon_snack"] > 0:
    afternoon_snack = recommend_snack(user_calorie_target * snack_ratios[user_goal]["afternoon_snack"],
                                      filtered_food_data, used_foods)
    used_calories += user_calorie_target * snack_ratios[user_goal]["afternoon_snack"]

dinner = recommend_meal(user_calorie_target * meal_ratios["dinner"], filtered_food_data, used_foods,
                        carb_ratio, protein_ratio, fat_ratio)
used_calories += user_calorie_target * meal_ratios["dinner"]

if snack_ratios[user_goal]["evening_snack"] > 0:
    evening_snack = recommend_snack(user_calorie_target * snack_ratios[user_goal]["evening_snack"],
                                    filtered_food_data, used_foods)
    used_calories += user_calorie_target * snack_ratios[user_goal]["evening_snack"]

# 결과 출력
print("추천 식단:")
print(f"아침: {', '.join(breakfast)}\n")
if snack_ratios[user_goal]["morning_snack"] > 0:
    print(f"간식 (아침-점심 사이): {morning_snack}\n")
print(f"점심: {', '.join(lunch)}\n")
if snack_ratios[user_goal]["afternoon_snack"] > 0:
    print(f"간식 (점심-저녁 사이): {afternoon_snack}\n")
print(f"저녁: {', '.join(dinner)}\n")
if snack_ratios[user_goal]["evening_snack"] > 0:
    print(f"간식 (저녁 이후): {evening_snack}\n")