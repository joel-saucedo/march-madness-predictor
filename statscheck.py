import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# For modeling
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, roc_curve,
                             auc, classification_report)
from sklearn.model_selection import train_test_split, learning_curve
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import OLSInfluence
from numpy.linalg import cond, eigvals

# =============================================================================
# 0. Setup Directories
# =============================================================================
os.makedirs("./plots", exist_ok=True)
os.makedirs("./results", exist_ok=True)

# =============================================================================
# 1. Load and Standardize Data
# =============================================================================
data_file = "./data/ncaa_2002_2024_cleaned.csv"
df = pd.read_csv(data_file)

# Standardize column names (replace spaces with underscores)
df.columns = df.columns.str.replace(" ", "_")

# Expected metrics (including all strength-of-schedule measures)
expected_metrics = [
    "NetRtg", "ORtg", "DRtg", "AdjT", "Luck",
    "Strength_of_Schedule_NetRtg", "Strength_of_Schedule_ORtg", "Strength_of_Schedule_DRtg",
    "NCSOS_NetRtg"
]

# Warn if expected columns for Team_A and Team_B are missing
for metric in expected_metrics:
    for team in ["Team_A", "Team_B"]:
        col_name = f"{team}_{metric}"
        if col_name not in df.columns:
            print(f"⚠️ Warning: Expected column '{col_name}' not found in data.")

# =============================================================================
# 2. Create Feature Differences
# =============================================================================
# For each metric, create a new column with the difference (Team_A - Team_B)
for metric in expected_metrics:
    colA = f"Team_A_{metric}"
    colB = f"Team_B_{metric}"
    diff_col = f"diff_{metric}"
    df[diff_col] = df[colA] - df[colB]

# List of feature difference columns for modeling
feature_cols = [f"diff_{metric}" for metric in expected_metrics]

# Drop rows with missing values in any of the features or the target ("Winner")
df_model = df.dropna(subset=feature_cols + ["Winner"]).copy()
df_model["Winner"] = pd.to_numeric(df_model["Winner"], errors="coerce")
df_model = df_model.dropna(subset=["Winner"])

# =============================================================================
# 3. Fit Logistic Regression Model (statsmodels)
# =============================================================================
X = df_model[feature_cols]
X_sm = sm.add_constant(X)  # add intercept
y = df_model["Winner"]

logit_model = sm.Logit(y, X_sm)
result = logit_model.fit(disp=0)  # MLE estimation; disp=0 to suppress iterative output
model_summary = result.summary2().as_text()

with open("./results/model_summary.txt", "w") as f:
    f.write(model_summary)
print("✅ Model summary saved to ./results/model_summary.txt")

# =============================================================================
# 4. Bias-Variance Decomposition via Learning Curves
# =============================================================================
# We use scikit-learn's learning_curve function to plot training vs. validation accuracy.
X_sk = df_model[feature_cols].values
y_sk = df_model["Winner"].values

train_sizes, train_scores, test_scores = learning_curve(
    LogisticRegression(max_iter=1000, solver='lbfgs'),
    X_sk, y_sk, cv=5, scoring='accuracy', n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 10)
)
train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)

plt.figure(figsize=(8, 6))
plt.plot(train_sizes, train_scores_mean, 'o-', label="Training Score")
plt.plot(train_sizes, test_scores_mean, 'o-', label="Validation Score")
plt.xlabel("Training Set Size")
plt.ylabel("Accuracy")
plt.title("Learning Curve: Bias-Variance Tradeoff")
plt.legend()
plt.tight_layout()
learning_curve_file = "./plots/learning_curve.png"
plt.savefig(learning_curve_file)
plt.close()
print(f"✅ Learning curve saved to {learning_curve_file}")

# =============================================================================
# 5. Hessian Analysis & Condition Number
# =============================================================================
# The Hessian matrix (second derivative of the log-likelihood) is computed as:
hessian = result.model.hessian(result.params)
# Compute the condition number of the Hessian (a measure of numerical stability)
hessian_cond = cond(hessian)
# Compute eigenvalues of the Hessian
eig_vals = eigvals(hessian)

plt.figure(figsize=(8, 6))
plt.plot(np.sort(np.abs(eig_vals)), marker='o')
plt.xlabel("Eigenvalue Index (sorted by magnitude)")
plt.ylabel("Absolute Eigenvalue")
plt.title("Eigenvalue Spectrum of the Hessian")
plt.tight_layout()
hessian_eig_file = "./plots/hessian_eigenvalues.png"
plt.savefig(hessian_eig_file)
plt.close()

with open("./results/hessian_condition.txt", "w") as f:
    f.write(f"Hessian Condition Number: {hessian_cond}\n")
    f.write("Eigenvalues:\n")
    f.write(np.array2string(eig_vals))
print(f"✅ Hessian condition and eigenvalues saved to ./results/hessian_condition.txt")

# =============================================================================
# 6. Regularization: L1 vs. L2
# =============================================================================
# We compare the coefficients from logistic regression models with L2 (default) and L1 penalties.
# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_sk, y_sk, test_size=0.3, random_state=42)

# L2 Regularization (Ridge)
lr_l2 = LogisticRegression(penalty='l2', solver='lbfgs', max_iter=1000)
lr_l2.fit(X_train, y_train)
coef_l2 = lr_l2.coef_.flatten()

# L1 Regularization (Lasso); note: using solver 'liblinear'
lr_l1 = LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000)
lr_l1.fit(X_train, y_train)
coef_l1 = lr_l1.coef_.flatten()

# Compare coefficients in a DataFrame
coef_df = pd.DataFrame({
    "Feature": feature_cols,
    "Coefficient_L2": coef_l2,
    "Coefficient_L1": coef_l1
})
coef_df_file = "./results/regularization_coefficients.csv"
coef_df.to_csv(coef_df_file, index=False)
print(f"✅ Regularization coefficients saved to {coef_df_file}")

# Plot coefficient comparison
plt.figure(figsize=(10, 6))
index = np.arange(len(feature_cols))
bar_width = 0.35
plt.bar(index, coef_l2, bar_width, label='L2')
plt.bar(index + bar_width, coef_l1, bar_width, label='L1')
plt.xticks(index + bar_width/2, feature_cols, rotation=45, ha='right')
plt.xlabel("Features")
plt.ylabel("Coefficient Value")
plt.title("Comparison of Regularization Coefficients (L1 vs L2)")
plt.legend()
plt.tight_layout()
coef_plot_file = "./plots/regularization_coefficients.png"
plt.savefig(coef_plot_file)
plt.close()
print(f"✅ Coefficient comparison plot saved to {coef_plot_file}")

# =============================================================================
# 7. Cook's Distance: Influence Diagnostics
# =============================================================================
# Using the statsmodels results, calculate Cook's distance for each observation.
influence = result.get_influence()
cooks_d = influence.cooks_distance[0]

plt.figure(figsize=(8, 6))
plt.stem(np.arange(len(cooks_d)), cooks_d, markerfmt=",")
plt.xlabel("Observation Index")
plt.ylabel("Cook's Distance")
plt.title("Cook's Distance for Each Observation")
plt.tight_layout()
cooks_plot_file = "./plots/cooks_distance.png"
plt.savefig(cooks_plot_file)
plt.close()

cooks_df = pd.DataFrame({"Observation": np.arange(len(cooks_d)), "Cooks_Distance": cooks_d})
cooks_df_file = "./results/cooks_distance.csv"
cooks_df.to_csv(cooks_df_file, index=False)
print(f"✅ Cook's distance plot saved to {cooks_plot_file}")
print(f"✅ Cook's distance values saved to {cooks_df_file}")

# =============================================================================
# 8. Additional: ROC Curve and Classification Metrics
# =============================================================================
# For further evaluation, we refit a logistic model using scikit-learn and plot the ROC curve.
lr_full = LogisticRegression(max_iter=1000)
lr_full.fit(X_train, y_train)
y_pred = lr_full.predict(X_test)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

y_proba = lr_full.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for Tournament Outcome Prediction")
plt.legend(loc="lower right")
plt.tight_layout()
roc_plot_file = "./plots/roc_curve.png"
plt.savefig(roc_plot_file)
plt.close()

with open("./results/classification_report.txt", "w") as f:
    f.write(f"Accuracy: {acc:.3f}\n\n")
    f.write("Confusion Matrix:\n")
    f.write(np.array2string(cm))
    f.write("\n\nClassification Report:\n")
    f.write(report)
print("✅ ROC curve and classification report saved to ./results/")

# =============================================================================
# 9. Save Final Model Dataset (if needed)
# =============================================================================
final_data_file = "./results/final_model_data.csv"
df_model.to_csv(final_data_file, index=False)
print(f"✅ Final model dataset saved to {final_data_file}")
