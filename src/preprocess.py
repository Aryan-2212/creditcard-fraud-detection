import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from datetime import datetime
from math import radians, sin, cos, sqrt, atan2

print("🚀 Loading dataset...")
df = pd.read_csv("data/raw/fraudTrain.csv")

# =========================
# DROP USELESS / LEAKAGE COLUMNS
# =========================
print("🧹 Dropping unnecessary columns...")

drop_cols = [
    'Unnamed: 0',   # index
    'cc_num',       # unique ID (leakage)
    'first',
    'last',
    'street',
    'trans_num',
    'unix_time'     # duplicate time info
]

df.drop(columns=drop_cols, inplace=True, errors='ignore')

# =========================
# FEATURE ENGINEERING
# =========================
print("⚙️ Feature engineering...")

# ---- Time Features ----
df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
df['hour'] = df['trans_date_trans_time'].dt.hour
df['day'] = df['trans_date_trans_time'].dt.day
df['month'] = df['trans_date_trans_time'].dt.month

# ---- Age ----
df['dob'] = pd.to_datetime(df['dob'])
df['age'] = (datetime.now() - df['dob']).dt.days // 365

# ---- Distance (Haversine) ----
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return 2 * R * atan2(sqrt(a), sqrt(1-a))

print("📍 Calculating distance...")
df['distance'] = df.apply(
    lambda x: haversine(x['lat'], x['long'], x['merch_lat'], x['merch_long']),
    axis=1
)

# ---- Drop original columns ----
df.drop(columns=['trans_date_trans_time', 'dob'], inplace=True)

# =========================
# ENCODING
# =========================
print("🔤 Encoding categorical features...")

for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# =========================
# SPLIT
# =========================
print("🔀 Splitting data...")

X = df.drop(columns=['is_fraud'])
y = df['is_fraud']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# =========================
# SCALING
# =========================
print("📏 Scaling features...")

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================
# SAVE
# =========================
print("💾 Saving processed data...")

with open("data/processed/data.pkl", "wb") as f:
    pickle.dump((X_train, X_test, y_train, y_test), f)

with open("data/processed/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("\n✅ PREPROCESSING COMPLETE")
print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)