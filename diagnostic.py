import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import roc_curve, auc, accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.linear_model import LogisticRegression

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, optimizers

# =============================================================================
# 0. Setup Directories
# =============================================================================
os.makedirs("./plots", exist_ok=True)
os.makedirs("./results", exist_ok=True)

# =============================================================================
# 1. Load Data and Create Features
# =============================================================================
data_file = "./data/ncaa_2002_2024_cleaned.csv"
df = pd.read_csv(data_file)

# Standardize column names
df.columns = df.columns.str.replace(" ", "_")

# Expected KenPom metrics (we use a subset for demonstration)
expected_metrics = [
    "NetRtg", "ORtg", "DRtg", "AdjT", "Luck", 
    "Strength_of_Schedule_NetRtg", "NCSOS_NetRtg"
]

# Create difference features for each metric: (Team_A - Team_B)
for metric in expected_metrics:
    df[f"diff_{metric}"] = df[f"Team_A_{metric}"] - df[f"Team_B_{metric}"]

# Our feature columns are the difference columns.
feature_cols = [f"diff_{metric}" for metric in expected_metrics]

# Keep only rows with complete feature and target data.
df_model = df.dropna(subset=feature_cols + ["Winner"]).copy()
df_model["Winner"] = pd.to_numeric(df_model["Winner"], errors="coerce")
df_model = df_model.dropna(subset=["Winner"])

# Separate features and target.
X = df_model[feature_cols].values
y = df_model["Winner"].values

# Split into train and test sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# =============================================================================
# 2. Build and Train the Final Neural Network Model
# =============================================================================
def build_model(params):
    dropout_rate = params["dropout_rate"]
    units = params["units"]
    l1_reg = params["l1_reg"]
    l2_reg = params["l2_reg"]
    use_second_layer = params["use_second_layer"]
    learning_rate = params["learning_rate"]
    
    model = models.Sequential()
    # Dropout layer applied to inputs
    model.add(layers.Dropout(dropout_rate, input_shape=(X_train.shape[1],)))
    # First Dense layer with L1 and L2 regularization
    model.add(layers.Dense(units, activation="relu", 
                           kernel_regularizer=regularizers.l1_l2(l1=l1_reg, l2=l2_reg)))
    if use_second_layer:
        # In our best configuration, we do not use a second layer.
        model.add(layers.Dense(16, activation="relu"))
    # Output layer for binary classification
    model.add(layers.Dense(1, activation="sigmoid"))
    
    optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
    return model

# Best hyperparameters from tuning:
best_params = {
    "dropout_rate": 0.5,
    "units": 48,
    "l1_reg": 0.0036039,
    "l2_reg": 0.0002406,
    "use_second_layer": False,
    "learning_rate": 0.0038519
}

final_model = build_model(best_params)

# Train the model
history = final_model.fit(X_train, y_train, epochs=17, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# =============================================================================
# 3. Evaluate Overfitting: Train vs. Validation Loss and ROC Curve
# =============================================================================
# Plot Train vs. Validation Loss
plt.figure(figsize=(8,6))
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Train vs. Validation Loss")
plt.legend()
plt.tight_layout()
plt.savefig("./plots/train_val_loss.png")
plt.close()
print("✅ Train vs. Validation loss plot saved to ./plots/train_val_loss.png")

# Compute predictions on test set
y_test_pred = final_model.predict(X_test).ravel()
test_accuracy = accuracy_score(y_test, (y_test_pred > 0.5).astype(int))
test_auc = roc_auc_score(y_test, y_test_pred)
print(f"Test Accuracy: {test_accuracy:.3f}")
print(f"Test AUC: {test_auc:.3f}")

# Plot ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_test_pred)
roc_auc_val = auc(fpr, tpr)
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc_val:.3f})")
plt.plot([0,1], [0,1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("./plots/roc_curve_final.png")
plt.close()
print("✅ ROC curve saved to ./plots/roc_curve_final.png")

# =============================================================================
# 4. Interpret Feature Importance via Learned Weights
# =============================================================================
# For our single hidden layer, extract weights from the first Dense layer.
dense_weights = final_model.layers[1].get_weights()[0]  # Shape: (n_features, units)
# Compute the average absolute weight for each input feature.
avg_abs_weights = np.mean(np.abs(dense_weights), axis=1)
feature_importance = pd.DataFrame({
    "Feature": feature_cols,
    "Avg_Abs_Weight": avg_abs_weights
}).sort_values(by="Avg_Abs_Weight", ascending=False)

feature_importance.to_csv("./results/feature_importance.csv", index=False)
print("✅ Feature importance saved to ./results/feature_importance.csv")

plt.figure(figsize=(8,6))
sns.barplot(data=feature_importance, x="Feature", y="Avg_Abs_Weight")
plt.xticks(rotation=45, ha="right")
plt.title("Average Absolute Weights per Feature")
plt.tight_layout()
plt.savefig("./plots/feature_importance.png")
plt.close()
print("✅ Feature importance plot saved to ./plots/feature_importance.png")

# =============================================================================
# 5. Statistical Analysis: Correlation Matrix of Input Features
# =============================================================================
corr_matrix = pd.DataFrame(X, columns=feature_cols).corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix of Feature Differences")
plt.tight_layout()
plt.savefig("./plots/feature_correlation_matrix.png")
plt.close()
print("✅ Feature correlation matrix plot saved to ./plots/feature_correlation_matrix.png")

# =============================================================================
# 6. Bias-Variance Estimation via Learning Curve (using Logistic Regression as proxy)
# =============================================================================
train_sizes, train_scores, test_scores = learning_curve(
    LogisticRegression(max_iter=1000),
    X, y, cv=5, scoring="accuracy", n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10)
)

train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)

plt.figure(figsize=(8,6))
plt.plot(train_sizes, train_scores_mean, 'o-', label="Training Score")
plt.plot(train_sizes, test_scores_mean, 'o-', label="CV Score")
plt.xlabel("Training Size")
plt.ylabel("Accuracy")
plt.title("Learning Curve (Logistic Regression Proxy)")
plt.legend(loc="best")
plt.tight_layout()
plt.savefig("./plots/learning_curve_logistic.png")
plt.close()
print("✅ Learning curve plot saved to ./plots/learning_curve_logistic.png")

# =============================================================================
# 7. Hessian Analysis: Approximate Hessian Condition Number of the Dense Layer
# =============================================================================
# For our dense layer, an approximation to the Hessian is given by the Gram matrix.
weights = final_model.layers[1].get_weights()[0]
hessian_approx = np.dot(weights.T, weights)
eigvals, _ = np.linalg.eig(hessian_approx)
condition_number = np.max(np.abs(eigvals)) / np.min(np.abs(eigvals))
print(f"Hessian Condition Number (approx): {condition_number}")

with open("./results/hessian_condition_number.txt", "w") as f:
    f.write(f"Hessian Condition Number (approx): {condition_number}\n")
    f.write("Eigenvalues:\n")
    f.write(np.array2string(eigvals))
print("✅ Hessian condition number saved to ./results/hessian_condition_number.txt")

# =============================================================================
# 8. Save Final Diagnostics Summary
# =============================================================================
with open("./results/neural_diagnostics.txt", "w") as f:
    f.write(f"Test Accuracy: {test_accuracy:.3f}\n")
    f.write(f"Test AUC: {test_auc:.3f}\n\n")
    f.write("Feature Importance:\n")
    f.write(feature_importance.to_string(index=False))
    f.write("\n\nHessian Condition Number (approx): {0}\n".format(condition_number))
    f.write("Eigenvalues:\n")
    f.write(np.array2string(eigvals))
print("✅ All diagnostics saved to ./results/neural_diagnostics.txt")
