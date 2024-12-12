import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import random
import json
from joblib import dump

# 시드값 설정
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)  # 시드값 설정

# 파일 경로 설정
input_path = '/content/drive/MyDrive/Colab Notebooks/P프데이터/Finish_FEBMI.csv'

# 1. 데이터 로드
input_data = pd.read_csv(input_path)

# Feature 선택
features = [
    'Age', 'Height', 'Weight', 'BMR', 'TDEE', 'BMI', 'TargetBMI', 
    'Calorie_Target', 'Calorie_Deficit', '총 운동시간', '하루소모칼로리', 
    '총 식사섭취 칼로리', 'ActivityLevel'
]

# 범주형 변수 (One-Hot Encoding 대상)
categorical_features = ['Gender', 'GoalType', 'preferred_body_part']

# 데이터 확인
assert all(col in input_data.columns for col in features), f"Missing features: {[col for col in features if col not in input_data.columns]}"
assert all(col in input_data.columns for col in categorical_features), f"Missing categorical features: {[col for col in categorical_features if col not in input_data.columns]}"

# One-Hot Encoding 처리
X = pd.get_dummies(input_data[features + categorical_features], columns=categorical_features).values

# Target 값 (예측 대상)
y = input_data['DaysToGoal'].fillna(0).values

# 데이터 스케일링
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Target 변수 로그 변환 (정규화)
y_log = np.log1p(y)

# 데이터셋 분할
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_log, test_size=0.2, random_state=42)

# 확인
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

# 5. PyTorch Dataset 정의
class GoalDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = GoalDataset(X_train, y_train)
test_dataset = GoalDataset(X_test, y_test)

# 6. 모델 정의
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
        layers.append(nn.Linear(in_dim, 1))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

# 7. 하이퍼파라미터 탐색
hidden_layer_combinations = [[256, 128, 64], [128, 64, 32]]
learning_rates = [0.01, 0.005]
batch_sizes = [128, 256]
early_stopping_patience = 7

best_mae = float('inf')
best_model = None
best_params = {}

for hidden_layers in hidden_layer_combinations:
    for lr in learning_rates:
        for batch_size in batch_sizes:
            fold_mae_scores = []
            kf = KFold(n_splits=5, shuffle=True, random_state=42)

            for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
                # Train/Validation split
                X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
                y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

                # PyTorch Dataset
                train_dataset_fold = GoalDataset(X_train_fold, y_train_fold)
                val_dataset_fold = GoalDataset(X_val_fold, y_val_fold)
                train_loader = DataLoader(train_dataset_fold, batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(val_dataset_fold, batch_size=batch_size, shuffle=False)

                # Initialize model
                model = FeedforwardNNImproved(input_dim=X_train.shape[1], hidden_layer_sizes=hidden_layers)
                optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)
                scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
                criterion = nn.MSELoss()

                # Early stopping setup
                best_loss = float('inf')
                patience_counter = 0

                # Training loop
                for epoch in range(50):
                    model.train()
                    for batch_X, batch_y in train_loader:
                        optimizer.zero_grad()
                        outputs = model(batch_X).squeeze()
                        loss = criterion(outputs, batch_y)
                        loss.backward()
                        optimizer.step()

                    # Validation
                    model.eval()
                    val_loss = 0
                    with torch.no_grad():
                        for batch_X, batch_y in val_loader:
                            outputs = model(batch_X).squeeze()
                            val_loss += criterion(outputs, batch_y).item()

                    # Early stopping
                    if val_loss < best_loss:
                        best_loss = val_loss
                        patience_counter = 0
                        best_fold_model = model
                    else:
                        patience_counter += 1
                        if patience_counter >= early_stopping_patience:
                            break

                # MAE 계산
                val_predictions, val_actuals = [], []
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        outputs = model(batch_X).squeeze()
                        val_predictions.extend(outputs.tolist())
                        val_actuals.extend(batch_y.tolist())

                val_predictions = np.expm1(val_predictions)
                val_actuals = np.expm1(val_actuals)
                mae = mean_absolute_error(val_actuals, val_predictions)
                fold_mae_scores.append(mae)

            avg_mae = np.mean(fold_mae_scores)
            if avg_mae < best_mae:
                best_mae = avg_mae
                best_model = best_fold_model
                best_params = {
                    'hidden_layers': hidden_layers,
                    'learning_rate': lr,
                    'batch_size': batch_size
                }

# 최적 하이퍼파라미터 출력
print(f"Best Parameters: {best_params}")
print(f"Cross-Validation Best MAE: {best_mae:.2f}")

# 8. 테스트 데이터 평가
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

test_predictions, test_actuals = [], []
best_model.eval()
with torch.no_grad():
    for batch_X, batch_y in test_loader:
        outputs = best_model(batch_X).squeeze()
        test_predictions.extend(outputs.tolist())
        test_actuals.extend(batch_y.tolist())

# 지수 변환하여 복원
test_predictions = np.expm1(test_predictions)
test_actuals = np.expm1(test_actuals)

# 평가 지표 계산
mae_test = mean_absolute_error(test_actuals, test_predictions)
mse_test = mean_squared_error(test_actuals, test_predictions)
r2_test = r2_score(test_actuals, test_predictions)

# 학습이 끝난 후 최적 모델 저장
save_path = "/content/drive/MyDrive/Colab Notebooks/P프로젝트/P_model.pth"
torch.save(best_model.state_dict(), save_path)
print(f"모델 가중치 저장 완료: {save_path}")

# 최종 결과 출력
print("\nTest Performance:")
print(f"Test MAE: {mae_test:.2f}")
print(f"Test MSE: {mse_test:.2f}")
print(f"Test R²: {r2_test:.2f}")

# 1. Feature 열 저장
feature_columns = list(pd.get_dummies(input_data[features + categorical_features], columns=categorical_features).columns)
feature_path = "/content/drive/MyDrive/Colab Notebooks/P프데이터/feature_columns.json"
with open(feature_path, 'w') as f:
    json.dump(feature_columns, f)
print(f"Feature 목록 저장 완료: {feature_path}")

# 2. Scaler 저장
scaler_path = "/content/drive/MyDrive/Colab Notebooks/P프데이터/scaler.joblib"
dump(scaler, scaler_path)
print(f"Scaler 저장 완료: {scaler_path}")

# 3. 모델 가중치 저장
model_path = "/content/drive/MyDrive/Colab Notebooks/P프데이터/P_model.pth"
torch.save(best_model.state_dict(), model_path)
print(f"모델 가중치 저장 완료: {model_path}")