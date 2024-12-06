import pandas as pd
import json

# 데이터 로드
food_data_path = '/Users/hong-yeonghun/Desktop/P프/final_food_data.csv'  # 식품 데이터 경로
food_data = pd.read_csv(food_data_path, encoding='utf-8')

# 식품 데이터 전처리
food_data['식품중량'] = food_data['식품중량'].str.replace('ml', 'g').str.replace('m', '').str.replace('g', '')
food_data['식품중량'] = pd.to_numeric(food_data['식품중량'], errors='coerce')
food_data = food_data.dropna(subset=['식품중량'])

# 음식 분류 함수
def classify_food(row):
    """
    음식 분류: 밥류, 국류, 반찬류, 디저트류 등으로 분류
    """
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
food_data['음식분류'] = food_data.apply(classify_food, axis=1)

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
    "저지방 고단백": {"carb_ratio": 0.4, "protein_ratio": 0.4, "fat_ratio": 0.2},
    "균형 식단": {"carb_ratio": 0.5, "protein_ratio": 0.3, "fat_ratio": 0.2},
    "벌크업": {"carb_ratio": 0.6, "protein_ratio": 0.3, "fat_ratio": 0.1},
}

# 식단 추천 함수
def recommend_diet(calorie_target, food_data, carb_target, protein_target, fat_target, used_foods):
    meal_ratios = {"breakfast": 0.3, "lunch": 0.35, "snack": 0.15, "dinner": 0.2}
    recommended_meals = {}

    for meal, ratio in meal_ratios.items():
        meal_calories = calorie_target * ratio
        meal_carb_target = carb_target * ratio
        meal_protein_target = protein_target * ratio
        meal_fat_target = fat_target * ratio

        if meal == "snack":
            snack_food = food_data[food_data['음식분류'] == '디저트류']
            snack = snack_food.sample(1).iloc[0]
            portion = min(meal_calories / snack['칼로리_100g'] * 100, 100)  # 최대 100g
            used_foods.append(snack['식품명'])
            recommended_meals[meal] = f"{snack['식품명']} {portion:.1f}g"
        else:
            rice_food = food_data[food_data['음식분류'] == '밥류']
            side_dish_food = food_data[food_data['음식분류'] == '반찬류']
            rice = rice_food.sample(1).iloc[0]
            side_dish = side_dish_food.sample(1).iloc[0]
            rice_portion = min(meal_calories * 0.6 / rice['칼로리_100g'] * 100, 300)
            side_portion = min(meal_calories * 0.4 / side_dish['칼로리_100g'] * 100, 200)
            used_foods.extend([rice['식품명'], side_dish['식품명']])
            recommended_meals[meal] = f"{rice['식품명']} {rice_portion:.1f}g / {side_dish['식품명']} {side_portion:.1f}g"

    return recommended_meals

# 사용자 맞춤 식단 추천
def get_custom_diet(user_info):
    current_bmr = calculate_bmr(user_info['current_weight'], user_info['height'], user_info['age'], user_info['gender'])
    target_bmr = calculate_bmr(user_info['target_weight'], user_info['height'], user_info['age'], user_info['gender'])
    current_tdee = calculate_tdee(current_bmr, user_info['activity_level'])
    target_tdee = calculate_tdee(target_bmr, user_info['activity_level'])
    goal = goal_ratios[user_info['goal_type']]

    carb_target = (target_tdee * goal['carb_ratio']) / 4
    protein_target = (target_tdee * goal['protein_ratio']) / 4
    fat_target = (target_tdee * goal['fat_ratio']) / 9

    return recommend_diet(target_tdee, food_data, carb_target, protein_target, fat_target, [])

# 사용자 입력
user_info = {
    "current_weight": 60,
    "target_weight": 65,
    "height": 170,
    "age": 25,
    "gender": "Male",
    "activity_level": 3,
    "goal_type": "벌크업",
}

result = get_custom_diet(user_info)
print(json.dumps(result, ensure_ascii=False, indent=4))
