import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, silhouette_score
from scipy.stats import randint, uniform
import numpy as np
import joblib
import json

# 1. 데이터 로드 및 전처리

# 모델 학습 데이터 로드 (목표 체중 도달 기간 예측)
data_path = 'goal_BMI.csv'  # 첫 번째 모델 데이터 경로
data = pd.read_csv(data_path)

# 칼로리 적자 계산 및 목표 유형 인코딩
data['Calorie_Deficit'] = data['TDEE'] - data['Calorie_Target']
goal_type_mapping = {'diet': 0, 'maintenance': 1, 'bulk-up': 2}
data['GoalTypeEncoded'] = data['GoalType'].map(goal_type_mapping)

# 입력 데이터 (X)와 출력 데이터 (y) 분리
X = data[['Age', 'Height', 'Weight', 'TargetWeight', 'TDEE', 'Calorie_Target',
          'ActivityLevel', 'Calorie_Deficit', 'BMI', 'TargetBMI', 'GoalTypeEncoded']]
y = data['DaysToGoal']

# 데이터 분할 (학습용/테스트용)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Gradient Boosting Regressor 하이퍼파라미터 튜닝
param_distributions = {
    'n_estimators': randint(100, 1000),  # 트리의 개수
    'learning_rate': uniform(0.01, 0.3),  # 학습률
    'max_depth': randint(2, 10),  # 트리 최대 깊이
    'min_samples_split': randint(2, 20),  # 내부 노드를 분할하기 위한 최소 샘플 수
    'min_samples_leaf': randint(1, 20),  # 리프 노드의 최소 샘플 수
    'subsample': uniform(0.5, 0.5),  # 트리 훈련 샘플 비율
    'max_features': ['sqrt', 'log2', None]  # 분할에 사용할 최대 특성 수
}

# RandomizedSearchCV를 이용한 하이퍼파라미터 튜닝
random_search = RandomizedSearchCV(
    estimator=GradientBoostingRegressor(random_state=42),
    param_distributions=param_distributions,
    n_iter=100,  # 테스트할 조합 수
    scoring='neg_mean_absolute_error',  # 평가 기준: 평균 절대 오차
    cv=10,  # 10-fold 교차 검증
    verbose=2,
    random_state=42,
    n_jobs=-1  # 병렬 처리
)

# 모델 학습
random_search.fit(X_train, y_train)
best_model = random_search.best_estimator_  # 최적의 모델
print(f"Best Parameters: {random_search.best_params_}")

# 학습된 모델 저장
model_filename = 'goal_prediction_model_tuned.pkl'
joblib.dump(best_model, model_filename)

# 3. 식품 데이터 로드 및 전처리
food_data_path = '/Users/hong-yeonghun/Desktop/P프/final_food_data.csv'
food_data = pd.read_csv(food_data_path, encoding='utf-8')

# 식품 중량 전처리 (숫자로 변환)
food_data['식품중량'] = food_data['식품중량'].str.replace(r'[^\d.]', '', regex=True)
food_data['식품중량'] = pd.to_numeric(food_data['식품중량'], errors='coerce')
food_data = food_data.dropna(subset=['식품중량'])  # 중량 값이 없는 데이터 제거

# 100g 기준 영양소 계산
food_data['칼로리_100g'] = (food_data['에너지(kcal)'] / food_data['식품중량']) * 100
food_data['탄수화물_100g'] = (food_data['탄수화물(g)'] / food_data['식품중량']) * 100
food_data['단백질_100g'] = (food_data['단백질(g)'] / food_data['식품중량']) * 100
food_data['지방_100g'] = (food_data['지방(g)'] / food_data['식품중량']) * 100

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
    elif any(x in row['식품대분류명'] for x in ["빵 및 과자류", "음료 및 차류", "유제품류 및 빙과류"]):
        return "디저트류"
    else:
        return "기타"

food_data['음식분류'] = food_data.apply(classify_food, axis=1)

# 4. K-Means 클러스터링
features = ['칼로리_100g', '탄수화물_100g', '단백질_100g', '지방_100g']
scaler = StandardScaler()
normalized_data = scaler.fit_transform(food_data[features])

# 적정 클러스터 개수 탐색 (실루엣 점수 기반)
best_k = 0
best_score = -1
for n_clusters in range(2, 10):  # 2~9개의 클러스터 탐색
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(normalized_data)
    score = silhouette_score(normalized_data, labels)
    if score > best_score:
        best_k = n_clusters
        best_score = score

print(f"Optimal number of clusters: {best_k}")

# 최적 클러스터 수로 K-Means 학습
kmeans = KMeans(n_clusters=best_k, random_state=42)
food_data['Cluster'] = kmeans.fit_predict(normalized_data)

# 목표별 영양소 비율 설정
goal_ratios = {
    "저지방 고단백": {"carb_ratio": 0.4, "protein_ratio": 0.4, "fat_ratio": 0.2},
    "균형 식단": {"carb_ratio": 0.5, "protein_ratio": 0.3, "fat_ratio": 0.2},
    "벌크업": {"carb_ratio": 0.6, "protein_ratio": 0.3, "fat_ratio": 0.1},
}

# 5. 식단 추천 함수
def recommend_diet(tdee, food_data, carb_target, protein_target, fat_target, used_foods):
    # 목표 클러스터 선택 (영양소 기준)
    target_cluster = kmeans.predict([[carb_target, protein_target, fat_target, tdee]])[0]
    recommendations = (
        food_data[(food_data['Cluster'] == target_cluster) & (~food_data['식품명'].isin(used_foods))]
        .head(5)
        [['식품명', '칼로리_100g', '탄수화물_100g', '단백질_100g', '지방_100g']]
        .to_dict(orient='records')
    )
    return recommendations

# 6. 최종 API 함수
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
    return {
        "diet_plan": diet_plan,
        "recommendations": recommendations,
        "predicted_goal_days": predicted_days
    }

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
print(json.dumps(result, ensure_ascii=False, indent=4))
