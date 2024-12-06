import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import json

# 데이터 로드
food_data_path = '/Users/hong-yeonghun/Desktop/P프/final_food_data.csv'  # 식품 데이터 경로
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

# 데이터 정규화
features = ['칼로리_100g', '탄수화물_100g', '단백질_100g', '지방_100g']
scaler = StandardScaler()
normalized_data = scaler.fit_transform(food_data[features])

# K-Means 군집화
kmeans = KMeans(n_clusters=5, random_state=42)  # 5개의 군집으로 분류
food_data['Cluster'] = kmeans.fit_predict(normalized_data)

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

# K-Means 기반 추천 함수
def recommend_by_cluster(nutrients):
    """
    주어진 영양소 목표값(칼로리, 탄수화물, 단백질, 지방)에 가장 가까운 클러스터에서 식품 추천.
    """
    nutrients_scaled = scaler.transform([nutrients])  # 입력값 정규화
    cluster = kmeans.predict(nutrients_scaled)[0]  # 클러스터 예측
    recommendations = food_data[food_data['Cluster'] == cluster].head(5)['식품명'].tolist()
    return recommendations

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

    used_foods = []
    diet_plan = recommend_diet(target_tdee, food_data, carb_target, protein_target, fat_target, used_foods)
    recommendations = recommend_by_cluster([target_tdee, carb_target, protein_target, fat_target])

    return {
        "diet_plan": diet_plan,
        "recommendations": recommendations
    }

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

# 결과 출력
result = get_custom_diet(user_info)
print(json.dumps(result, ensure_ascii=False, indent=4))
