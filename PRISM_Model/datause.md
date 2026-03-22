
#### 使用的ACS数据
1. Education
pct_less_than_9th_grade
2. Language
pct_lep
3. Race& Ethinicity
pct_hispanic
pct_non_hispanic_black
4. Age 
pct_senior
pct_young_adults   # 可选
5. SES 
pct_below_poverty
pct_unemployed
pct_uninsured
per_capita_income
pct_female_headed_households
6. Housing & Transportation
pct_overcrowded_housing
pct_households_without_a_vehicle
pct_work_at_home
7. Occupation
pct_service

#### Mobility
median_non_home_dwell_time_lag4
non_home_ratio_lag4
full_time_work_behavior_devices_lag4

#### Target
Weekly_New_Cases_per_100k

###
X_train/X_test 现在是 21 列（zipcode + Week_Start_Date + 19 个特征）。


### 
训练目标： 给定一个 zipcode 的 mobility + 社会结构（static）时间序列，
预测该 zipcode 对应时间段的 COVID 序列

我先在需要创建训练数据，写入loader.py文件

模型写入model.py文件：
使用两个模型结构LSTM 和 Transformer
(B,T,F) → (B,T,H) → (B,T,1)
1️⃣ 输入（X）
假设：
90 个训练 zipcode
每个 zipcode 有 T 周（比如 20）
每周 F 个特征（比如 3 mobility + 14 static = 17）

输入X∈(B,T,F)  B是batch size，T是时间步，F是特征维度

2️⃣ 输出（y）
要预测：
每个时间步的 COVID 值  y∈RB×T×1

训练写入单独的main.py

结构：
mobility sequence (B,T,F_dyn)
        │
        ▼
      LSTM
        │
        ▼
   dyn_hidden (B,T,H)

static ACS (B,F_static)
        │
        ▼
       MLP
        │
        ▼
 static_hidden (B,Hs)
        │
   unsqueeze + repeat
        ▼
 static_context (B,T,Hs)

dyn_hidden + static_context
        │
      concat
        ▼
   fusion (B,T,H+Hs)
        │
      Linear
        ▼
   y_pred (B,T,1)


结构代码
  - loader.py: 每个样本现在返回 (x_dyn, x_static, y)，x_static 只取第一行（ACS 在 zipcode 内恒定）
  - model.py:
    - mobility → LSTM → dyn_hidden (B,T,H)
    - ACS → MLP → static_hidden (B,Hs) → unsqueeze+expand → (B,T,Hs)
    - concat → Linear → y_pred (B,T,1)
  - config.py: 新增 DYNAMIC_COLS / STATIC_COLS 定义，新增 static_hidden_size 参数
  - 参数量从 55,105 降到 5,169，train R² 从 0.23 升到 0.33