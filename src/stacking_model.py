import os
import pickle
import joblib

from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

# =========================
# CREATE MODELS FOLDER
# =========================
os.makedirs("models", exist_ok=True)

print("📥 Loading processed data...")

with open("data/processed/data.pkl", "rb") as f:
    X_train, X_test, y_train, y_test = pickle.load(f)

# =========================
# BASE MODELS
# =========================
estimators = [
    ("lr", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ("dt", DecisionTreeClassifier(class_weight="balanced", random_state=42)),
    ("rf", RandomForestClassifier(n_estimators=100, n_jobs=-1, class_weight="balanced", random_state=42))
]

# =========================
# STACKING MODEL
# =========================
print("🚀 Training Stacking Model...")

model = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(),
    n_jobs=-1
)

model.fit(X_train, y_train)

# =========================
# SAVE MODEL
# =========================
joblib.dump(model, "models/stacking_model.pkl")
print("✅ Model saved successfully")

# =========================
# PREDICTIONS
# =========================
print("📊 Evaluating model...")

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# =========================
# METRICS
# =========================
print("\n===== STACKING MODEL RESULTS =====")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nROC-AUC Score:")
print(roc_auc_score(y_test, y_prob))