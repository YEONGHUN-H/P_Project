from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from joblib import load
import json

app = FastAPI()

# 사용자 입력 데이터 모델
class UserInfo(BaseModel):
    username: str
    current_weight: float
    target_weight: float
    height: float
    age: int
    gender: str
    activity_level: int  # 1~5 사이 값
    goal_type: str  # 저지방 고단백, 균형 식단, 벌크업
    preferred_body_part: str  # 선호 운동 부위
    bmr: float
    tdee: float
    bmi: float
    target_bmi: float

# 모델 및 스케일러 로드
feature_path = "/content/drive/MyDrive/Colab Notebooks/P프데이터/feature_columns.json"
scaler_path = "/content/drive/MyDrive/Colab Notebooks/P프데이터/scaler.joblib"
model_path = "/content/drive/MyDrive/Colab Notebooks/P프데이터/P_model.pth"

with open(feature_path, 'r') as f:
    expected_columns = json.load(f)

scaler = load(scaler_path)

# PyTorch 모델 정의 (학습 시와 동일해야 함)
class FeedforwardNNImproved(nn.Module):
    def __init__(self, input_dim, hidden_layer_sizes):
        super(FeedforwardNNImproved, self).__init__()
        layers = []
        in_dim = input_dim
        for size in hidden_layer_sizes:
            layers.append(nn.Linear(in_dim, size))
            layers.append(nn.BatchNorm1d(size))
            layers.append(nn.LeakyReLU())
            layers.append(nn.Dropout(0.2))
            in_dim = size
        layers.append(nn.Linear(in_dim, 1))  # 마지막 출력 레이어
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

hidden_layer_sizes = [128, 64, 32]
input_dim = len(expected_columns)
model = FeedforwardNNImproved(input_dim, hidden_layer_sizes)
model.load_state_dict(torch.load(model_path))
model.eval()

@app.post("/predict")
def predict_goal_duration(user_info: UserInfo):
    # 입력 데이터 처리
    input_data = pd.DataFrame([{
        "Age": user_info.age,
        "Height": user_info.height / 100,  # cm → m
        "Weight": user_info.current_weight,
        "TargetWeight": user_info.target_weight,
        "BMR": user_info.bmr,
        "TDEE": user_info.tdee,
        "BMI": user_info.bmi,
        "TargetBMI": user_info.target_bmi,
        "Calorie_Target": user_info.tdee - 500,  # 예시 칼로리 목표
        "Calorie_Deficit": 500,
        "총 운동시간": 120,
        "하루소모칼로리": 400,
        "총 식사섭취 칼로리": 2000,
        "ActivityLevel": user_info.activity_level,
        "Gender": user_info.gender,
        "GoalType": user_info.goal_type,
        "preferred_body_part": user_info.preferred_body_part
    }])

    # 범주형 변수 처리
    categorical_features = ['Gender', 'GoalType', 'preferred_body_part']
    input_data = pd.get_dummies(input_data, columns=categorical_features)

    # 누락된 열 추가 및 정렬
    input_data = input_data.reindex(columns=expected_columns, fill_value=0)

    # 데이터 스케일링
    X_input = scaler.transform(input_data)

    # 모델 예측
    with torch.no_grad():
        X_tensor = torch.tensor(X_input, dtype=torch.float32)
        prediction = model(X_tensor).item()
        days_to_goal = np.expm1(prediction)  # 로그 변환 복원

    # 결과 반환
    return {
        "username": user_info.username,
        "days_to_goal": round(days_to_goal, 2),  # 목표 달성 예상 기간 (일수)
        "message": f"{user_info.username}님의 목표 달성까지 예상 소요 기간은 약 {round(days_to_goal, 2)}일입니다."
    }
