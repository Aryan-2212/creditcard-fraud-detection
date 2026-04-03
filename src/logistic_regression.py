import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

print("📥 Loading processed data...")

with open("data/processed/data.pkl", "rb") as f:
    X_train, X_test, y_train, y_test = pickle.load(f)

# =========================
# MODEL
# =========================
print("🚀 Training Logistic Regression...")

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    n_jobs=-1
)

model.fit(X_train, y_train)

# =========================
# PREDICTIONS
# =========================
print("📊 Evaluating model...")

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# =========================
# METRICS
# =========================
print("\n===== LOGISTIC REGRESSION RESULTS =====")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nROC-AUC Score:")
print(roc_auc_score(y_test, y_prob))