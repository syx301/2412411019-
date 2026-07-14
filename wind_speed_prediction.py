"""
2026年商学院机器学习课程项目 - 风速序列预测
完整可运行代码
包含：数据预处理、特征工程、3种模型训练与对比、可视化、模型保存(.pth)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 配置
# ============================================================
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(OUTPUT_DIR, 'figures')
MODEL_DIR = os.path.join(OUTPUT_DIR, 'models')
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 数据路径（请根据实际路径修改）
DATA_PATH = r"D:/AAA新建文件夹/机器学习/WindSpeed_merged_full.csv"
TARGET_COL = 'Speed_100m'
WINDOW_SIZE = 48  # 8小时 = 48个10分钟时间步

# ============================================================
# 1. 数据加载与预处理
# ============================================================
print("=" * 60)
print("1. 数据加载与预处理")
print("=" * 60)

df = pd.read_csv(DATA_PATH, encoding='utf-8-sig')
df.columns = ['Timestamp', 'Speed_10m', 'Speed_50m', 'Speed_100m', 'SpeedMax',
              'DirectionAvg', 'TemperatureAvg', 'TemperatureMax',
              'PressureAvg', 'PressureMax', 'HumidityAvg', 'HumidityMax']
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df = df.sort_values('Timestamp').reset_index(drop=True)

print(f"数据集大小: {df.shape}")
print(f"时间范围: {df['Timestamp'].min()} ~ {df['Timestamp'].max()}")

# 异常值处理（IQR方法）
feature_cols = ['Speed_10m', 'Speed_50m', 'Speed_100m', 'SpeedMax',
                'DirectionAvg', 'TemperatureAvg', 'TemperatureMax',
                'PressureAvg', 'PressureMax', 'HumidityAvg', 'HumidityMax']

for col in feature_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outlier_count = ((df[col] < lower) | (df[col] > upper)).sum()
    if outlier_count > 0:
        print(f"  {col}: {outlier_count} 个异常值 → 边界值替换")
        df[col] = df[col].clip(lower, upper)

# 删除重复时间戳
dup_count = df['Timestamp'].duplicated().sum()
if dup_count > 0:
    df = df.drop_duplicates(subset='Timestamp', keep='first').reset_index(drop=True)
    print(f"删除重复时间戳: {dup_count} 条")

print(f"清洗后数据集大小: {df.shape}")

# ========== 可视化: 特征分布图 ==========
fig, axes = plt.subplots(3, 4, figsize=(18, 12))
axes = axes.flatten()
for i, col in enumerate(feature_cols):
    axes[i].hist(df[col], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    axes[i].set_title(f'{col} 分布', fontsize=11)
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('频数')
for j in range(len(feature_cols), 12):
    axes[j].axis('off')
plt.suptitle('各特征变量分布图', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'feature_distribution.png'), dpi=150)
plt.close()

# ========== 可视化: 风速时间序列 ==========
fig, ax = plt.subplots(figsize=(16, 5))
ax.plot(df['Timestamp'], df['Speed_10m'], label='10m风速', alpha=0.7)
ax.plot(df['Timestamp'], df['Speed_50m'], label='50m风速', alpha=0.7)
ax.plot(df['Timestamp'], df['Speed_100m'], label='100m风速', alpha=0.7)
ax.set_xlabel('时间')
ax.set_ylabel('风速 (m/s)')
ax.set_title('不同高度风速时间序列')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'wind_speed_timeseries.png'), dpi=150)
plt.close()

# ========== 可视化: 相关性热力图 ==========
corr_cols = ['Speed_10m', 'Speed_50m', 'Speed_100m', 'SpeedMax',
             'DirectionAvg', 'TemperatureAvg', 'PressureAvg', 'HumidityAvg']
corr_matrix = df[corr_cols].corr()
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
            square=True, linewidths=0.5, ax=ax)
ax.set_title('特征相关性热力图')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'correlation_heatmap.png'), dpi=150)
plt.close()

# ============================================================
# 2. 特征工程
# ============================================================
print("\n" + "=" * 60)
print("2. 特征工程")
print("=" * 60)

# 时间周期特征
df['Hour'] = df['Timestamp'].dt.hour
df['Month'] = df['Timestamp'].dt.month
df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)

# 滞后特征
for lag in [1, 2, 3, 6, 12, 48]:
    df[f'Speed_100m_lag{lag}'] = df[TARGET_COL].shift(lag)
    df[f'Speed_10m_lag{lag}'] = df['Speed_10m'].shift(lag)
    df[f'Speed_50m_lag{lag}'] = df['Speed_50m'].shift(lag)

# 滚动统计特征
for window in [6, 12, 48]:
    df[f'Speed_100m_roll_mean_{window}'] = df[TARGET_COL].rolling(window).mean()
    df[f'Speed_100m_roll_std_{window}'] = df[TARGET_COL].rolling(window).std()

# 差分特征
df['Speed_100m_diff1'] = df[TARGET_COL].diff(1)
df['Speed_100m_diff6'] = df[TARGET_COL].diff(6)

df = df.dropna().reset_index(drop=True)
print(f"特征工程后数据集大小: {df.shape}")

INPUT_COLS = [col for col in df.columns if col not in
              ['Timestamp', 'TemperatureMax', 'PressureMax', 'HumidityMax', TARGET_COL, 'Hour', 'Month']]
INPUT_DIM = len(INPUT_COLS)
print(f"输入特征数量: {INPUT_DIM}")

# ============================================================
# 3. 数据划分与标准化 (7:2:1)
# ============================================================
print("\n" + "=" * 60)
print("3. 数据划分 (7:2:1)")
print("=" * 60)

n_total = len(df)
n_train = int(n_total * 0.7)
n_val = int(n_total * 0.2)
n_test = n_total - n_train - n_val

train_df = df.iloc[:n_train]
val_df = df.iloc[n_train:n_train + n_val]
test_df = df.iloc[n_train + n_val:]

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(train_df[INPUT_COLS].values)
y_train_scaled = scaler_y.fit_transform(train_df[TARGET_COL].values.reshape(-1, 1)).flatten()

X_val_scaled = scaler_X.transform(val_df[INPUT_COLS].values)
y_val_scaled = scaler_y.transform(val_df[TARGET_COL].values.reshape(-1, 1)).flatten()

X_test_scaled = scaler_X.transform(test_df[INPUT_COLS].values)
y_test_scaled = scaler_y.transform(test_df[TARGET_COL].values.reshape(-1, 1)).flatten()

print(f"训练集: {len(train_df)} ({len(train_df)/n_total:.1%})")
print(f"验证集: {len(val_df)} ({len(val_df)/n_total:.1%})")
print(f"测试集: {len(test_df)} ({len(test_df)/n_total:.1%})")

# ============================================================
# 4. 滑动窗口数据构建
# ============================================================
print("\n" + "=" * 60)
print("4. 滑动窗口数据构建")
print("=" * 60)

def create_windows(X, y, window_size, pred_steps):
    X_windows, y_windows = [], []
    for i in range(len(X) - window_size - pred_steps + 1):
        X_windows.append(X[i:i + window_size])
        y_windows.append(y[i + window_size:i + window_size + pred_steps])
    return np.array(X_windows), np.array(y_windows)

# 单步预测 (pred_steps=1)
X_train_1, y_train_1 = create_windows(X_train_scaled, y_train_scaled, WINDOW_SIZE, 1)
X_val_1, y_val_1 = create_windows(X_val_scaled, y_val_scaled, WINDOW_SIZE, 1)
X_test_1, y_test_1 = create_windows(X_test_scaled, y_test_scaled, WINDOW_SIZE, 1)

# 多步预测A (pred_steps=6, 1小时)
X_train_6, y_train_6 = create_windows(X_train_scaled, y_train_scaled, WINDOW_SIZE, 6)
X_val_6, y_val_6 = create_windows(X_val_scaled, y_val_scaled, WINDOW_SIZE, 6)
X_test_6, y_test_6 = create_windows(X_test_scaled, y_test_scaled, WINDOW_SIZE, 6)

# 多步预测B (pred_steps=96, 16小时)
X_train_96, y_train_96 = create_windows(X_train_scaled, y_train_scaled, WINDOW_SIZE, 96)
X_val_96, y_val_96 = create_windows(X_val_scaled, y_val_scaled, WINDOW_SIZE, 96)
X_test_96, y_test_96 = create_windows(X_test_scaled, y_test_scaled, WINDOW_SIZE, 96)

print(f"单步: train={X_train_1.shape}, val={X_val_1.shape}, test={X_test_1.shape}")
print(f"多步A: train={X_train_6.shape}, val={X_val_6.shape}, test={X_test_6.shape}")
print(f"多步B: train={X_train_96.shape}, val={X_val_96.shape}, test={X_test_96.shape}")

# ============================================================
# 5. 模型定义
# ============================================================
print("\n" + "=" * 60)
print("5. 模型定义")
print("=" * 60)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=1, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1), :])

class TransformerModel(nn.Module):
    def __init__(self, input_size, d_model=64, nhead=4, num_layers=2, output_size=1, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                    dim_feedforward=d_model * 4,
                                                    dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, output_size)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        return self.fc(x[:, -1, :])

# ============================================================
# 6. 训练与评估函数
# ============================================================

def train_deep_model(model, X_train_w, y_train_w, X_val_w, y_val_w,
                     epochs=50, batch_size=128, lr=0.001, model_name='model', task_suffix=''):
    """训练深度学习模型"""
    X_t = torch.FloatTensor(X_train_w).to(device)
    y_t = torch.FloatTensor(y_train_w).to(device)
    if y_t.ndim == 1:
        y_t = y_t.unsqueeze(1)
    X_v = torch.FloatTensor(X_val_w).to(device)
    y_v = torch.FloatTensor(y_val_w).to(device)
    if y_v.ndim == 1:
        y_v = y_v.unsqueeze(1)

    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    save_path = os.path.join(MODEL_DIR, f'{model_name}{task_suffix}.pth')

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg_train_loss = epoch_loss / len(loader)
        train_losses.append(avg_train_loss)

        model.eval()
        with torch.no_grad():
            val_pred = model(X_v)
            val_loss = criterion(val_pred, y_v).item()
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs} | Train: {avg_train_loss:.6f} | Val: {val_loss:.6f}")

    # 绘制loss曲线
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_losses, label='训练损失')
    ax.plot(val_losses, label='验证损失')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title(f'{model_name}{task_suffix} 训练过程')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f'{model_name}_loss{task_suffix}.png'), dpi=150)
    plt.close()

    # 加载最佳模型
    model.load_state_dict(torch.load(save_path, weights_only=True))
    return model


def evaluate_model(y_true, y_pred, pred_steps=1, model_name='', task_suffix=''):
    """评估模型并绘制结果图"""
    # 反标准化
    y_true_orig = scaler_y.inverse_transform(y_true.reshape(-1, 1)).flatten()
    y_pred_orig = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()

    mse = mean_squared_error(y_true_orig, y_pred_orig)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true_orig, y_pred_orig)
    r2 = r2_score(y_true_orig, y_pred_orig)

    print(f"  {model_name}{task_suffix}: MSE={mse:.4f} RMSE={rmse:.4f} MAE={mae:.4f} R²={r2:.4f}")

    # 预测对比图
    n = min(500, len(y_true_orig))
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(range(n), y_true_orig[:n], label='真实值', linewidth=1.5)
    ax.plot(range(n), y_pred_orig[:n], label='预测值', linewidth=1.5, alpha=0.8)
    ax.set_xlabel('时间步')
    ax.set_ylabel('风速 (m/s)')
    ax.set_title(f'{model_name}{task_suffix} 预测结果对比')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f'{model_name}_pred{task_suffix}.png'), dpi=150)
    plt.close()

    # 真实值vs预测值散点图
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true_orig, y_pred_orig, alpha=0.3, s=10)
    vmin = min(y_true_orig.min(), y_pred_orig.min())
    vmax = max(y_true_orig.max(), y_pred_orig.max())
    ax.plot([vmin, vmax], [vmin, vmax], 'r--', linewidth=2, label='理想拟合线')
    ax.set_xlabel('真实值 (m/s)')
    ax.set_ylabel('预测值 (m/s)')
    ax.set_title(f'{model_name}{task_suffix} 真实值 vs 预测值')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f'{model_name}_scatter{task_suffix}.png'), dpi=150)
    plt.close()

    return {'MSE': float(round(mse, 4)), 'RMSE': float(round(rmse, 4)),
            'MAE': float(round(mae, 4)), 'R2': float(round(r2, 4))}


# ============================================================
# 7. 实验执行 - 单步预测
# ============================================================
print("\n" + "=" * 60)
print("7. 单步预测实验")
print("=" * 60)

SINGLE_OUTPUT = 1

# --- Linear Regression ---
print("\n--- Linear Regression (单步) ---")
lr_model_1 = LinearRegression()
lr_model_1.fit(X_train_1.reshape(X_train_1.shape[0], -1), y_train_1.flatten())
y_pred_lr1 = lr_model_1.predict(X_test_1.reshape(X_test_1.shape[0], -1))
metrics_lr_1 = evaluate_model(y_test_1.flatten(), y_pred_lr1, 1, 'LR', '_single')

# --- LSTM ---
print("\n--- LSTM (单步) ---")
lstm_model_1 = LSTMModel(INPUT_DIM, hidden_size=64, num_layers=2,
                          output_size=SINGLE_OUTPUT, dropout=0.2).to(device)
lstm_model_1 = train_deep_model(lstm_model_1, X_train_1, y_train_1, X_val_1, y_val_1,
                                 epochs=50, batch_size=128, lr=0.001,
                                 model_name='LSTM', task_suffix='_single')
lstm_model_1.eval()
with torch.no_grad():
    y_pred_lstm1 = lstm_model_1(torch.FloatTensor(X_test_1).to(device)).cpu().numpy().flatten()
metrics_lstm_1 = evaluate_model(y_test_1.flatten(), y_pred_lstm1, 1, 'LSTM', '_single')

# --- Transformer ---
print("\n--- Transformer (单步) ---")
trans_model_1 = TransformerModel(INPUT_DIM, d_model=64, nhead=4, num_layers=2,
                                  output_size=SINGLE_OUTPUT, dropout=0.1).to(device)
trans_model_1 = train_deep_model(trans_model_1, X_train_1, y_train_1, X_val_1, y_val_1,
                                  epochs=50, batch_size=128, lr=0.001,
                                  model_name='Transformer', task_suffix='_single')
trans_model_1.eval()
with torch.no_grad():
    y_pred_trans1 = trans_model_1(torch.FloatTensor(X_test_1).to(device)).cpu().numpy().flatten()
metrics_trans_1 = evaluate_model(y_test_1.flatten(), y_pred_trans1, 1, 'Transformer', '_single')


# ============================================================
# 8. 实验执行 - 多步预测A (1h = 6步)
# ============================================================
print("\n" + "=" * 60)
print("8. 多步预测A (8h历史 → 1h未来, 6步)")
print("=" * 60)

MULTI_A_OUTPUT = 6

# --- Linear Regression ---
print("\n--- Linear Regression (多步A) ---")
lr_model_6 = LinearRegression()
lr_model_6.fit(X_train_6.reshape(X_train_6.shape[0], -1), y_train_6)
y_pred_lr6 = lr_model_6.predict(X_test_6.reshape(X_test_6.shape[0], -1))
metrics_lr_6 = evaluate_model(y_test_6, y_pred_lr6, MULTI_A_OUTPUT, 'LR', '_multiA')

# --- LSTM ---
print("\n--- LSTM (多步A) ---")
lstm_model_6 = LSTMModel(INPUT_DIM, hidden_size=128, num_layers=3,
                          output_size=MULTI_A_OUTPUT, dropout=0.2).to(device)
lstm_model_6 = train_deep_model(lstm_model_6, X_train_6, y_train_6, X_val_6, y_val_6,
                                 epochs=50, batch_size=128, lr=0.001,
                                 model_name='LSTM', task_suffix='_multiA')
lstm_model_6.eval()
with torch.no_grad():
    y_pred_lstm6 = lstm_model_6(torch.FloatTensor(X_test_6).to(device)).cpu().numpy()
metrics_lstm_6 = evaluate_model(y_test_6, y_pred_lstm6, MULTI_A_OUTPUT, 'LSTM', '_multiA')

# --- Transformer ---
print("\n--- Transformer (多步A) ---")
trans_model_6 = TransformerModel(INPUT_DIM, d_model=64, nhead=4, num_layers=2,
                                  output_size=MULTI_A_OUTPUT, dropout=0.1).to(device)
trans_model_6 = train_deep_model(trans_model_6, X_train_6, y_train_6, X_val_6, y_val_6,
                                  epochs=50, batch_size=128, lr=0.001,
                                  model_name='Transformer', task_suffix='_multiA')
trans_model_6.eval()
with torch.no_grad():
    y_pred_trans6 = trans_model_6(torch.FloatTensor(X_test_6).to(device)).cpu().numpy()
metrics_trans_6 = evaluate_model(y_test_6, y_pred_trans6, MULTI_A_OUTPUT, 'Transformer', '_multiA')


# ============================================================
# 9. 实验执行 - 多步预测B (16h = 96步)
# ============================================================
print("\n" + "=" * 60)
print("9. 多步预测B (8h历史 → 16h未来, 96步)")
print("=" * 60)

MULTI_B_OUTPUT = 96

# --- Linear Regression ---
print("\n--- Linear Regression (多步B) ---")
lr_model_96 = LinearRegression()
lr_model_96.fit(X_train_96.reshape(X_train_96.shape[0], -1), y_train_96)
y_pred_lr96 = lr_model_96.predict(X_test_96.reshape(X_test_96.shape[0], -1))
metrics_lr_96 = evaluate_model(y_test_96, y_pred_lr96, MULTI_B_OUTPUT, 'LR', '_multiB')

# --- LSTM ---
print("\n--- LSTM (多步B) ---")
lstm_model_96 = LSTMModel(INPUT_DIM, hidden_size=64, num_layers=2,
                           output_size=MULTI_B_OUTPUT, dropout=0.2).to(device)
lstm_model_96 = train_deep_model(lstm_model_96, X_train_96, y_train_96, X_val_96, y_val_96,
                                  epochs=30, batch_size=128, lr=0.001,
                                  model_name='LSTM', task_suffix='_multiB')
lstm_model_96.eval()
with torch.no_grad():
    y_pred_lstm96 = lstm_model_96(torch.FloatTensor(X_test_96).to(device)).cpu().numpy()
metrics_lstm_96 = evaluate_model(y_test_96, y_pred_lstm96, MULTI_B_OUTPUT, 'LSTM', '_multiB')

# --- Transformer ---
print("\n--- Transformer (多步B) ---")
trans_model_96 = TransformerModel(INPUT_DIM, d_model=64, nhead=4, num_layers=2,
                                   output_size=MULTI_B_OUTPUT, dropout=0.1).to(device)
trans_model_96 = train_deep_model(trans_model_96, X_train_96, y_train_96, X_val_96, y_val_96,
                                  epochs=30, batch_size=128, lr=0.001,
                                  model_name='Transformer', task_suffix='_multiB')
trans_model_96.eval()
with torch.no_grad():
    y_pred_trans96 = trans_model_96(torch.FloatTensor(X_test_96).to(device)).cpu().numpy()
metrics_trans_96 = evaluate_model(y_test_96, y_pred_trans96, MULTI_B_OUTPUT, 'Transformer', '_multiB')


# ============================================================
# 10. 综合对比可视化
# ============================================================
print("\n" + "=" * 60)
print("10. 模型综合对比")
print("=" * 60)

results = {
    '单步预测': {'Linear Regression': metrics_lr_1, 'LSTM': metrics_lstm_1, 'Transformer': metrics_trans_1},
    '多步预测A(1h)': {'Linear Regression': metrics_lr_6, 'LSTM': metrics_lstm_6, 'Transformer': metrics_trans_6},
    '多步预测B(16h)': {'Linear Regression': metrics_lr_96, 'LSTM': metrics_lstm_96, 'Transformer': metrics_trans_96},
}

for task_name, task_results in results.items():
    print(f"\n{task_name}:")
    for model_name, m in task_results.items():
        print(f"  {model_name}: MSE={m['MSE']} RMSE={m['RMSE']} MAE={m['MAE']} R²={m['R2']}")

# ========== 综合对比柱状图 ==========
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
task_labels = ['单步预测', '多步预测A(1h)', '多步预测B(16h)']
model_labels = ['Linear Regression', 'LSTM', 'Transformer']

for idx, (task_key, task_label) in enumerate(zip(results.keys(), task_labels)):
    mse_vals = [results[task_key][m]['MSE'] for m in model_labels]
    rmse_vals = [results[task_key][m]['RMSE'] for m in model_labels]
    mae_vals = [results[task_key][m]['MAE'] for m in model_labels]
    r2_vals = [results[task_key][m]['R2'] for m in model_labels]

    x = np.arange(len(model_labels))
    width = 0.2
    ax = axes[idx]
    ax.bar(x - 1.5 * width, mse_vals, width, label='MSE', color='#e74c3c')
    ax.bar(x - 0.5 * width, rmse_vals, width, label='RMSE', color='#3498db')
    ax.bar(x + 0.5 * width, mae_vals, width, label='MAE', color='#2ecc71')
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, fontsize=8)
    ax.set_title(task_label)
    ax.legend(fontsize=8)

    ymax = max(max(mse_vals), max(rmse_vals), max(mae_vals))
    for i, r2 in enumerate(r2_vals):
        ax.text(i, ymax * 0.85, f'R²={r2:.3f}', ha='center', fontsize=9,
                fontweight='bold', color='#8e44ad')

plt.suptitle('三种模型在不同预测任务上的性能对比', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'model_comparison.png'), dpi=150)
plt.close()

# ========== 预测序列对比图 ==========
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 单步预测对比
n = min(300, len(metrics_lr_1.get('y_true_orig', [0])))
# 重新获取原始值
yt1 = scaler_y.inverse_transform(y_test_1.flatten().reshape(-1, 1)).flatten()
yp_lr1 = scaler_y.inverse_transform(y_pred_lr1.reshape(-1, 1)).flatten()
yp_lstm1 = scaler_y.inverse_transform(y_pred_lstm1.reshape(-1, 1)).flatten()
yp_trans1 = scaler_y.inverse_transform(y_pred_trans1.reshape(-1, 1)).flatten()

n = min(300, len(yt1))
axes[0].plot(range(n), yt1[:n], 'k-', label='真实值', linewidth=1.5)
axes[0].plot(range(n), yp_lr1[:n], label='LR', alpha=0.6)
axes[0].plot(range(n), yp_lstm1[:n], label='LSTM', alpha=0.6)
axes[0].plot(range(n), yp_trans1[:n], label='Transformer', alpha=0.6)
axes[0].set_title('单步预测 - 三模型对比')
axes[0].set_xlabel('时间步')
axes[0].set_ylabel('风速 (m/s)')
axes[0].legend()

# 多步A样本对比
sample = 50
yt6_s = scaler_y.inverse_transform(y_test_6[sample].reshape(-1, 1)).flatten()
yp_lr6_s = scaler_y.inverse_transform(y_pred_lr6[sample].reshape(-1, 1)).flatten()
yp_lstm6_s = scaler_y.inverse_transform(y_pred_lstm6[sample].reshape(-1, 1)).flatten()
yp_trans6_s = scaler_y.inverse_transform(y_pred_trans6[sample].reshape(-1, 1)).flatten()

time_steps = np.arange(1, 7) * 10
axes[1].plot(time_steps, yt6_s, 'ko-', label='真实值', linewidth=2)
axes[1].plot(time_steps, yp_lr6_s, 's--', label='LR', alpha=0.7)
axes[1].plot(time_steps, yp_lstm6_s, '^--', label='LSTM', alpha=0.7)
axes[1].plot(time_steps, yp_trans6_s, 'v--', label='Transformer', alpha=0.7)
axes[1].set_title('多步预测A (1h) - 样本对比')
axes[1].set_xlabel('预测时间 (分钟)')
axes[1].set_ylabel('风速 (m/s)')
axes[1].legend()

# 多步B样本对比
sample = 30
yt96_s = scaler_y.inverse_transform(y_test_96[sample].reshape(-1, 1)).flatten()
yp_lr96_s = scaler_y.inverse_transform(y_pred_lr96[sample].reshape(-1, 1)).flatten()
yp_lstm96_s = scaler_y.inverse_transform(y_pred_lstm96[sample].reshape(-1, 1)).flatten()
yp_trans96_s = scaler_y.inverse_transform(y_pred_trans96[sample].reshape(-1, 1)).flatten()

time_steps_b = np.arange(1, 97) * 10
axes[2].plot(time_steps_b, yt96_s, label='真实值', linewidth=1.5)
axes[2].plot(time_steps_b, yp_lr96_s, label='LR', alpha=0.6)
axes[2].plot(time_steps_b, yp_lstm96_s, label='LSTM', alpha=0.6)
axes[2].plot(time_steps_b, yp_trans96_s, label='Transformer', alpha=0.6)
axes[2].set_title('多步预测B (16h) - 样本对比')
axes[2].set_xlabel('预测时间 (分钟)')
axes[2].set_ylabel('风速 (m/s)')
axes[2].legend()

plt.suptitle('不同预测任务下的模型预测序列对比', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'prediction_sequence_comparison.png'), dpi=150)
plt.close()

# ========== R²对比图 ==========
fig, ax = plt.subplots(figsize=(10, 5))
tasks = list(results.keys())
x = np.arange(len(tasks))
width = 0.25

for i, model_name in enumerate(model_labels):
    r2_vals = [results[task][model_name]['R2'] for task in tasks]
    ax.bar(x + i * width, r2_vals, width, label=model_name)

ax.set_xticks(x + width)
ax.set_xticklabels(tasks)
ax.set_ylabel('R²')
ax.set_title('不同任务下各模型R²对比')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'r2_comparison.png'), dpi=150)
plt.close()

# 保存结果指标
import json
with open(os.path.join(OUTPUT_DIR, 'results_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("\n" + "=" * 60)
print("全部实验完成！")
print(f"图表保存在: {FIG_DIR}")
print(f"模型保存在: {MODEL_DIR} (.pth格式)")
print("=" * 60)
