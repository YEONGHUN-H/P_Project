import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import joblib
import json

# 모델 학습 데이터 로드 (예시용)
data_path = 'goal_BMI.csv'  # 첫 번째 모델 데이터 경로
data = pd.read_csv(data_path)

# 첫 번째 모델 전처리
data['Calorie_Deficit'] = data['TDEE'] - data['Calorie_Target']
goal_type_mapping = {'diet': 0, 'maintenance': 1, 'bulk-up': 2}
data['GoalTypeEncoded'] = data['GoalType'].map(goal_type_mapping)

X = data[['Age', 'Height', 'Weight', 'TargetWeight', 'TDEE', 'Calorie_Target',
          'ActivityLevel', 'Calorie_Deficit', 'BMI', 'TargetBMI', 'GoalTypeEncoded']]
y = data['DaysToGoal']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 첫 번째 모델 학습
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.05, 0.1, 0.2],
    'max_depth': [3, 4, 5],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

gb_model = GradientBoostingRegressor(random_state=42)
grid_search = GridSearchCV(estimator=gb_model, param_grid=param_grid, cv=5, scoring='neg_mean_absolute_error', verbose=1)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

# 첫 번째 모델 저장
model_filename = 'goal_prediction_model_tuned.pkl'
joblib.dump(best_model, model_filename)

# 식단 추천 데이터 로드
food_data_path = '/Users/hong-yeonghun/Desktop/P프/final_food_data.csv'
food_data = pd.read_csv(food_data_path, encoding='utf-8')

# 식단 추천 전처리
food_data['식품중량'] = food_data['식품중량'].str.replace('ml', 'g').str.replace('m', '').str.replace('g', '')
food_data['식품중량'] = pd.to_numeric(food_data['식품중량'], errors='coerce')
food_data = food_data.dropna(subset=['식품중량'])

food_data['칼로리_100g'] = (food_data['에너지(kcal)'] / food_data['식품중량']) * 100
food_data['탄수화물_100g'] = (food_data['탄수화물(g)'] / food_data['식품중량']) * 100
food_data['단백질_100g'] = (food_data['단백질(g)'] / food_data['식품중량']) * 100
food_data['지방_100g'] = (food_data['지방(g)'] / food_data['식품중량']) * 100

# 음식 분류
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
    elif any(x in row['식품대분류명'] for x in ["빵 및 과자류", "음료 및 차류", "유제품류 및 빙과류"]):
        return "디저트류"
    else:
        return "기타"

food_data['음식분류'] = food_data.apply(classify_food, axis=1)

# K-Means 학습
features = ['칼로리_100g', '탄수화물_100g', '단백질_100g', '지방_100g']
scaler = StandardScaler()
normalized_data = scaler.fit_transform(food_data[features])

kmeans = KMeans(n_clusters=5, random_state=42)
food_data['Cluster'] = kmeans.fit_predict(normalized_data)

# 목표별 영양소 비율
goal_ratios = {
    "저지방 고단백": {"carb_ratio": 0.4, "protein_ratio": 0.4, "fat_ratio": 0.2},
    "균형 식단": {"carb_ratio": 0.5, "protein_ratio": 0.3, "fat_ratio": 0.2},
    "벌크업": {"carb_ratio": 0.6, "protein_ratio": 0.3, "fat_ratio": 0.1},
}

# 사용자 맞춤 식단 추천
def recommend_diet(tdee, food_data, carb_target, protein_target, fat_target, used_foods):
    recommendations = food_data[food_data['Cluster'] == 0].head(5)['식품명'].tolist()
    return recommendations

# 최종 API 함수
def get_combined_output(user_info):
    # 첫 번째 모델: 목표 체중 도달 예측
    input_df = pd.DataFrame([{
        "Age": user_info["age"],
        "Height": user_info["height"],
        "Weight": user_info["current_weight"],
        "TargetWeight": user_info["target_weight"],
        "TDEE": user_info["tdee"],
        "Calorie_Target": user_info["calorie_target"],
        "ActivityLevel": user_info["activity_level"],
        "Calorie_Deficit": user_info["tdee"] - user_info["calorie_target"],
        "BMI": user_info["bmi"],
        "TargetBMI": user_info["target_bmi"],
        "GoalTypeEncoded": goal_type_mapping[user_info["goal_type"]]
    }])
    predicted_days = int(best_model.predict(input_df)[0])

    # 두 번째 모델: 식단 추천
    goal = goal_ratios[user_info["goal_type"]]
    carb_target = (user_info["tdee"] * goal['carb_ratio']) / 4
    protein_target = (user_info["tdee"] * goal['protein_ratio']) / 4
    fat_target = (user_info["tdee"] * goal['fat_ratio']) / 9

    diet_plan = recommend_diet(user_info["tdee"], food_data, carb_target, protein_target, fat_target, [])
    recommendations = recommend_diet(user_info["tdee"], food_data, carb_target, protein_target, fat_target, [])

    # JSON 반환
    return json.dumps({
        "diet_plan": diet_plan,
        "recommendations": recommendations,
        "predicted_goal_days": f"성공 예측일: {predicted_days}일"
    }, ensure_ascii=False, indent=4)

# 사용자 입력 예시
user_info = {
    "current_weight": 70,
    "target_weight": 65,
    "height": 175,
    "age": 30,
    "gender": "Male",
    "activity_level": 3,
    "tdee": 2200,
    "calorie_target": 1800,
    "bmi": 22.8,
    "target_bmi": 21.2,
    "goal_type": "저지방 고단백"
}

# 결과 출력
result = get_combined_output(user_info)
print(result)
