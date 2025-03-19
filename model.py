import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# For modeling
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, classification_report
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

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

# Standardize column names: remove spaces (e.g., "Strength of Schedule NetRtg" → "Strength_of_Schedule_NetRtg")
df.columns = df.columns.str.replace(" ", "_")

# Expected KenPom metrics (all of them, including strength-of-schedule measures)
expected_metrics = [
    "NetRtg", "ORtg", "DRtg", "AdjT", "Luck", 
    "Strength_of_Schedule_NetRtg", "Strength_of_Schedule_ORtg", "Strength_of_Schedule_DRtg",
    "NCSOS_NetRtg"
]

# (Optionally, print a warning if any expected column is missing from either team)
for metric in expected_metrics:
    for team in ["Team_A", "Team_B"]:
        col_name = f"{team}_{metric}"
        if col_name not in df.columns:
            print(f"⚠️ Warning: Expected column '{col_name}' not found in data.")

# =============================================================================
# 2. Create Feature Differences
# =============================================================================
# For each metric, compute the difference (Team_A - Team_B)
for metric in expected_metrics:
    colA = f"Team_A_{metric}"
    colB = f"Team_B_{metric}"
    diff_col = f"diff_{metric}"
    df[diff_col] = df[colA] - df[colB]

# Define the list of feature difference columns we will use for the model.
feature_cols = [f"diff_{metric}" for metric in expected_metrics]

# Drop rows with missing values in these features or in the target column "Winner"
df_model = df.dropna(subset=feature_cols + ["Winner"]).copy()

# Ensure the target variable is numeric (should be 0 or 1)
df_model["Winner"] = pd.to_numeric(df_model["Winner"], errors="coerce")
df_model = df_model.dropna(subset=["Winner"])

# =============================================================================
# 3. Fit Logistic Regression Model Using statsmodels (for full summary)
# =============================================================================
X = df_model[feature_cols]
X = sm.add_constant(X)  # adds an intercept
y = df_model["Winner"]

logit_model = sm.Logit(y, X)
result = logit_model.fit(disp=0)  # disp=0 to suppress output
model_summary = result.summary2().as_text()

with open("./results/model_summary.txt", "w") as f:
    f.write(model_summary)
print("✅ Model summary saved to ./results/model_summary.txt")

# =============================================================================
# 4. Evaluate Model Using scikit-learn
# =============================================================================
# For evaluation purposes, we use a train-test split.
X_sklearn = df_model[feature_cols]
y_sklearn = df_model["Winner"]

X_train, X_test, y_train, y_test = train_test_split(X_sklearn, y_sklearn, test_size=0.3, random_state=42)
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Logistic Regression Test Accuracy: {acc:.3f}")

# Generate confusion matrix and classification report.
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

with open("./results/classification_report.txt", "w") as f:
    f.write(f"Accuracy: {acc:.3f}\n\n")
    f.write("Confusion Matrix:\n")
    f.write(np.array2string(cm))
    f.write("\n\nClassification Report:\n")
    f.write(report)
print("✅ Classification report saved to ./results/classification_report.txt")

# =============================================================================
# 5. ROC Curve
# =============================================================================
y_proba = lr.predict_proba(X_test)[:, 1]
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
print(f"✅ ROC curve saved to {roc_plot_file}")

# =============================================================================
# 6. Scatter Plots: Feature Differences vs. Predicted Probability
# =============================================================================
# Add predicted probability (of Team_A winning) to the dataset for visualization.
df_model["predicted_proba"] = lr.predict_proba(X_sklearn)[:, 1]

for metric in expected_metrics:
    diff_col = f"diff_{metric}"
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df_model, x=diff_col, y="predicted_proba", hue="Winner", palette="coolwarm", alpha=0.7)
    plt.xlabel(f"Difference in {metric} (Team_A - Team_B)")
    plt.ylabel("Predicted Probability of Team A Winning")
    plt.title(f"{metric} Difference vs. Predicted Win Probability")
    plt.tight_layout()
    scatter_file = f"./plots/scatter_{metric}.png"
    plt.savefig(scatter_file)
    plt.close()
    print(f"✅ Scatter plot for {metric} saved to {scatter_file}")

# =============================================================================
# 7. Time Evolution Analysis: Average Feature Differences Over the Years
# =============================================================================
years = sorted(df_model["Year"].unique())
avg_diffs = {metric: [] for metric in expected_metrics}

for year in years:
    year_df = df_model[df_model["Year"] == year]
    for metric in expected_metrics:
        diff_col = f"diff_{metric}"
        avg_diffs[metric].append(year_df[diff_col].mean())

plt.figure(figsize=(10, 6))
for metric in expected_metrics:
    plt.plot(years, avg_diffs[metric], marker="o", label=f"Avg diff {metric}")
plt.xlabel("Year")
plt.ylabel("Average Difference (Team_A - Team_B)")
plt.title("Time Evolution of Average Feature Differences")
plt.legend()
plt.tight_layout()
time_plot_file = "./plots/time_evolution.png"
plt.savefig(time_plot_file)
plt.close()
print(f"✅ Time evolution plot saved to {time_plot_file}")

# =============================================================================
# 8. Save Final Dataset Used for Model Calibration
# =============================================================================
final_data_file = "./results/final_model_data.csv"
df_model.to_csv(final_data_file, index=False)
print(f"✅ Final model dataset saved to {final_data_file}")
