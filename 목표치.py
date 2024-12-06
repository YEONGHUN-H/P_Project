import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import joblib
import json

# 1. 데이터 로드
data_path = 'goal_BMI.csv'  # 가공된 데이터 파일 경로
data = pd.read_csv(data_path)

# 2. 특성 생성: 칼로리 적자 또는 잉여 계산
data['Calorie_Deficit'] = data['TDEE'] - data['Calorie_Target']

# 3. 입력 데이터(X)와 출력 데이터(y) 설정
X = data[['Age', 'Height', 'Weight', 'TargetWeight', 'TDEE', 'Calorie_Target',
          'ActivityLevel', 'Calorie_Deficit', 'BMI', 'TargetBMI']]
y = data['DaysToGoal']  # 목표 도달 일수

# 4. 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. 모델 학습 (Gradient Boosting Regressor)
model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=4, random_state=42)
model.fit(X_train, y_train)

# 6. 모델 평가
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error (MAE): {mae:.2f} days")

# 7. 모델 저장
model_filename = 'goal_prediction_model.pkl'
joblib.dump(model, model_filename)
print(f"Model saved to {model_filename}")

# 8. 예측 함수 정의
def predict_goal_days(user_data):
    input_df = pd.DataFrame([user_data])
    input_df['Calorie_Deficit'] = input_df['TDEE'] - input_df['Calorie_Target']  # 칼로리 적자 계산
    predicted_days = int(model.predict(input_df)[0])  # 예측값 반환
    
    # JSON 형식으로 결과 생성
    result = {
        "UserInput": user_data,
        "PredictedDaysToGoal": predicted_days
    }
    return json.dumps(result, ensure_ascii=False, indent=4)

# 9. 사용자 입력 예시
user_input = {
    'Age': 30,
    'Height': 175,
    'Weight': 80,
    'TargetWeight': 70,
    'TDEE': 2200,
    'Calorie_Target': 1700,  # 다이어트 예시
    'ActivityLevel': 3,
    'BMI': 26.1,
    'TargetBMI': 22.9
}

# 예측 실행
result_json = predict_goal_days(user_input)
print(result_json)
