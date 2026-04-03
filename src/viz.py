import matplotlib.pyplot as plt
import numpy as np

# =========================
# MODEL COMPARISON
# =========================

models = ["LR", "DT", "RF", "GB", "Ada", "Stack"]

recall = [0.77, 0.82, 0.74, 0.63, 0.36, 0.82]
precision = [0.08, 0.83, 0.98, 0.82, 0.81, 0.95]

x = np.arange(len(models))

plt.figure()
plt.bar(x - 0.2, recall, 0.4, label="Recall")
plt.bar(x + 0.2, precision, 0.4, label="Precision")
plt.xticks(x, models)
plt.title("Structured Dataset - Model Comparison")
plt.xlabel("Models")
plt.ylabel("Score")
plt.legend()
plt.show()