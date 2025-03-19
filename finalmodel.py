import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc

# =============================================================================
# 0. Setup Directories
# =============================================================================
os.makedirs("./final/plots", exist_ok=True)
os.makedirs("./final/results", exist_ok=True)

# =============================================================================
# 1. Load Data and Compute Feature Differences
# =============================================================================
data_file = "./data/ncaa_2002_2024_cleaned.csv"
df = pd.read_csv(data_file)

# Standardize column names (replace spaces with underscores)
df.columns = df.columns.str.replace(" ", "_")

# Define the KenPom-based metrics
metrics = [
    "NetRtg", "ORtg", "DRtg", "AdjT", "Luck", 
    "Strength_of_Schedule_NetRtg", "Strength_of_Schedule_ORtg", 
    "Strength_of_Schedule_DRtg", "NCSOS_NetRtg"
]

# Compute feature differences (Team_A - Team_B)
for metric in metrics:
    team_a_col = f"Team_A_{metric}"
    team_b_col = f"Team_B_{metric}"
    diff_col = f"diff_{metric}"
    if team_a_col in df.columns and team_b_col in df.columns:
        df[diff_col] = df[team_a_col] - df[team_b_col]
    else:
        print(f"⚠️ Missing columns: {team_a_col} or {team_b_col} - Dropping {diff_col}")

# List of difference feature names
feature_cols = [f"diff_{metric}" for metric in metrics]

# Drop rows with missing values in these features or the target "Winner"
df_model = df.dropna(subset=feature_cols + ["Winner"]).copy()
df_model["Winner"] = pd.to_numeric(df_model["Winner"], errors="coerce")
df_model = df_model.dropna(subset=["Winner"])

# =============================================================================
# 2. Correlation Analysis & Feature Selection via PCA/Univariate Predictiveness
# =============================================================================
# Compute the correlation matrix among the difference features.
corr_matrix = df_model[feature_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix of Feature Differences")
plt.tight_layout()
plt.savefig("./final/plots/correlation_matrix.png")
plt.close()

# For each pair of features with absolute correlation above a threshold,
# remove the one with lower absolute correlation with the target.
corr_threshold = 0.9
to_remove = set()
removal_log = []
for i in range(len(feature_cols)):
    for j in range(i+1, len(feature_cols)):
        feat_i = feature_cols[i]
        feat_j = feature_cols[j]
        corr_val = abs(corr_matrix.loc[feat_i, feat_j])
        if corr_val > corr_threshold:
            # Calculate absolute Pearson correlation with the target variable "Winner"
            corr_i = abs(df_model[feat_i].corr(df_model["Winner"]))
            corr_j = abs(df_model[feat_j].corr(df_model["Winner"]))
            if corr_i < corr_j:
                to_remove.add(feat_i)
                removal_log.append(f"Removed {feat_i} (|corr with target|={corr_i:.3f}) vs {feat_j} (|corr with target|={corr_j:.3f}), pair corr={corr_val:.3f}")
            else:
                to_remove.add(feat_j)
                removal_log.append(f"Removed {feat_j} (|corr with target|={corr_j:.3f}) vs {feat_i} (|corr with target|={corr_i:.3f}), pair corr={corr_val:.3f}")

selected_features = [feat for feat in feature_cols if feat not in to_remove]

# Save feature selection log
with open("./final/results/feature_selection.txt", "w") as f:
    f.write("Feature Selection Log based on high correlation and target predictiveness:\n")
    for line in removal_log:
        f.write(line + "\n")
    f.write("\nSelected Features for Modeling:\n")
    for feat in selected_features:
        f.write(feat + "\n")

# =============================================================================
# 3. Elastic Net Logistic Regression with Hyperparameter Tuning
# =============================================================================
# Standardize the selected features.
scaler = StandardScaler()
X_sklearn = scaler.fit_transform(df_model[selected_features])
y_sklearn = df_model["Winner"]

# Set up grid search parameters for Elastic Net regularization.
param_grid = {
    "C": [0.01, 0.1, 1, 10, 100],
    "l1_ratio": [0.1, 0.5, .9]
}
# Note: penalty='elasticnet' requires solver='saga'
elastic_net_lr = LogisticRegression(penalty="elasticnet", solver="saga", max_iter=5000)
grid_search = GridSearchCV(elastic_net_lr, param_grid, cv=5, scoring="accuracy")
grid_search.fit(X_sklearn, y_sklearn)

best_params = grid_search.best_params_
best_score = grid_search.best_score_

with open("./final/results/elastic_net_params.txt", "w") as f:
    f.write("Best Elastic Net Logistic Regression Parameters:\n")
    f.write(str(best_params) + "\n")
    f.write(f"Cross-validated Accuracy: {best_score:.3f}\n")

# Fit the final Elastic Net model on the full dataset.
final_model = LogisticRegression(penalty="elasticnet", solver="saga", max_iter=5000, **best_params)
final_model.fit(X_sklearn, y_sklearn)
y_pred = final_model.predict(X_sklearn)
acc = accuracy_score(y_sklearn, y_pred)
cm = confusion_matrix(y_sklearn, y_pred)
class_report = classification_report(y_sklearn, y_pred)

with open("./final/results/model_performance.txt", "w") as f:
    f.write(f"Elastic Net Logistic Regression Full Data Performance:\nAccuracy: {acc:.3f}\n")
    f.write("Confusion Matrix:\n")
    f.write(np.array2string(cm))
    f.write("\n\nClassification Report:\n")
    f.write(class_report)

# ROC Curve
y_proba = final_model.predict_proba(X_sklearn)[:, 1]
fpr, tpr, thresholds = roc_curve(y_sklearn, y_proba)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for Elastic Net Logistic Regression")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("./final/plots/elastic_net_roc_curve.png")
plt.close()

# =============================================================================
# 4. Hessian Analysis & Condition Number using statsmodels
# =============================================================================
# For detailed diagnostics, we refit a logistic regression using statsmodels.
X_sm = scaler.fit_transform(df_model[selected_features])
X_sm = sm.add_constant(X_sm)
logit_model = sm.Logit(y_sklearn, X_sm)
result = logit_model.fit(disp=0)

# Calculate Hessian matrix, its eigenvalues and condition number.
hessian = result.model.hessian(result.params)
eigvals = np.linalg.eigvals(hessian)
condition_number = np.max(np.abs(eigvals)) / np.min(np.abs(eigvals))

with open("./final/results/hessian_diagnostics.txt", "w") as f:
    f.write(f"Hessian Condition Number: {condition_number}\n")
    f.write("Eigenvalues of Hessian:\n")
    f.write(np.array2string(eigvals))

# =============================================================================
# 5. Cook's Distance: Identify Influential Points
# =============================================================================
influence = result.get_influence()
cooks_d = influence.cooks_distance[0]

plt.figure(figsize=(10, 6))
plt.stem(np.arange(len(cooks_d)), cooks_d, markerfmt=",")
plt.xlabel("Observation Index")
plt.ylabel("Cook's Distance")
plt.title("Cook's Distance for Logistic Regression")
plt.tight_layout()
plt.savefig("./final/plots/cooks_distance.png")
plt.close()

# Define an outlier threshold (common rule: Cook's distance > 4/n)
n = len(df_model)
cooks_threshold = 4 / n
outlier_indices = np.where(cooks_d > cooks_threshold)[0]

with open("./final/results/outliers.txt", "w") as f:
    f.write(f"Number of observations: {n}\n")
    f.write(f"Cook's distance threshold (4/n): {cooks_threshold:.6f}\n")
    f.write(f"Number of outliers identified: {len(outlier_indices)}\n")
    f.write("Indices of outliers:\n")
    f.write(np.array2string(outlier_indices))

# =============================================================================
# 6. Remove Outliers and Refit the Model
# =============================================================================
df_model_clean = df_model.drop(df_model.index[outlier_indices])
X_clean = scaler.fit_transform(df_model_clean[selected_features])
y_clean = df_model_clean["Winner"]

# Refit the Elastic Net model on the cleaned data.
final_model_clean = LogisticRegression(penalty="elasticnet", solver="saga", max_iter=5000, **best_params)
final_model_clean.fit(X_clean, y_clean)
y_pred_clean = final_model_clean.predict(X_clean)
acc_clean = accuracy_score(y_clean, y_pred_clean)
cm_clean = confusion_matrix(y_clean, y_pred_clean)
class_report_clean = classification_report(y_clean, y_pred_clean)

with open("./final/results/model_performance_clean.txt", "w") as f:
    f.write(f"Model Performance After Removing Outliers:\nAccuracy: {acc_clean:.3f}\n")
    f.write("Confusion Matrix:\n")
    f.write(np.array2string(cm_clean))
    f.write("\n\nClassification Report:\n")
    f.write(class_report_clean)

# Save the final cleaned dataset.
df_model_clean.to_csv("./final/final_model_data_cleaned.csv", index=False)

print("✅ All diagnostics and model refinements completed. Check the './final' directory for outputs.")




# =============================================================================
# 7. Quick Fix for Convergence Warning: Increase max_iter and adjust tolerance
# =============================================================================
# (For example, refit using a higher max_iter and a lower tolerance.)
final_model_fix = LogisticRegression(penalty="elasticnet", solver="saga", max_iter=10000, tol=1e-4, **best_params)
final_model_fix.fit(X_sklearn, y_sklearn)
print("✅ Model refit with increased max_iter and adjusted tol.")

# =============================================================================
# 8. Bootstrap Analysis: Assessing Variability of Model Estimates
# =============================================================================
from sklearn.utils import resample

n_bootstraps = 1000  # number of bootstrap samples
bootstrap_scores = []

for i in range(n_bootstraps):
    # Sample with replacement from the training data
    X_boot, y_boot = resample(X_sklearn, y_sklearn, random_state=i)
    model_boot = LogisticRegression(penalty="elasticnet", solver="saga", max_iter=10000, tol=1e-4, **best_params)
    model_boot.fit(X_boot, y_boot)
    score = model_boot.score(X_sklearn, y_sklearn)  # use full data for simplicity
    bootstrap_scores.append(score)

bootstrap_scores = np.array(bootstrap_scores)
plt.figure(figsize=(8, 6))
plt.hist(bootstrap_scores, bins=20, edgecolor="k", alpha=0.7)
plt.xlabel("Bootstrap Accuracy")
plt.ylabel("Frequency")
plt.title("Bootstrap Distribution of Model Accuracy")
plt.tight_layout()
plt.savefig("./final/plots/bootstrap_accuracy.png")
plt.close()

with open("./final/results/bootstrap_summary.txt", "w") as f:
    f.write("Bootstrap Analysis Summary:\n")
    f.write(f"Mean Accuracy: {bootstrap_scores.mean():.3f}\n")
    f.write(f"Std Accuracy: {bootstrap_scores.std():.3f}\n")

print("✅ Bootstrap analysis complete. Results saved to ./final/results/bootstrap_summary.txt and plot to ./final/plots/bootstrap_accuracy.png")

# =============================================================================
# 9. Learning Curves: Plot to Check Overfitting vs. Underfitting
# =============================================================================
from sklearn.model_selection import learning_curve

train_sizes, train_scores, test_scores = learning_curve(
    LogisticRegression(penalty="elasticnet", solver="saga", max_iter=10000, tol=1e-4, **best_params),
    X_sklearn,
    y_sklearn,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 10),
    random_state=42
)

train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)

plt.figure(figsize=(8, 6))
plt.plot(train_sizes, train_scores_mean, 'o-', label="Training Score")
plt.plot(train_sizes, test_scores_mean, 'o-', label="Cross-Validation Score")
plt.xlabel("Training Set Size")
plt.ylabel("Accuracy")
plt.title("Learning Curve")
plt.legend(loc="best")
plt.tight_layout()
plt.savefig("./final/plots/learning_curve.png")
plt.close()

print("✅ Learning curve plot saved to ./final/plots/learning_curve.png")

# =============================================================================
# 10. Extended Cross-Validation: Repeated CV for Robust Model Evaluation
# =============================================================================
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
cv_scores = cross_val_score(
    LogisticRegression(penalty="elasticnet", solver="saga", max_iter=10000, tol=1e-4, **best_params),
    X_sklearn,
    y_sklearn,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1
)

with open("./final/results/extended_cv.txt", "w") as f:
    f.write("Extended Cross-Validation Results:\n")
    f.write(f"Mean Accuracy: {cv_scores.mean():.3f}\n")
    f.write(f"Std Accuracy: {cv_scores.std():.3f}\n")

print("✅ Extended cross-validation complete. Results saved to ./final/results/extended_cv.txt")

# =============================================================================
# 11. Final Model Diagnostics Recap: Save Model Coefficients and Diagnostics
# =============================================================================
coef_array = np.concatenate((final_model_fix.intercept_.flatten(), final_model_fix.coef_.flatten()))
coef_df = pd.DataFrame({
    "Feature": ["const"] + selected_features,
    "Coefficient": coef_array
})
coef_df.to_csv("./final/results/model_coefficients.csv", index=False)

print("✅ Model coefficients saved to ./final/results/model_coefficients.csv")
