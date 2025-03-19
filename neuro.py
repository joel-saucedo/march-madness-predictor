"""
neural_model.py

This script performs the following steps:
1. Loads the cleaned tournament and KenPom dataset from ./data/ncaa_2002_2024_cleaned.csv.
2. Computes the correlation matrix and performs PCA on the feature differences.
   - Identifies highly correlated features (above a threshold) and drops one from each pair.
   - Saves the correlation heatmap and PCA explained variance plot.
3. Prepares the dataset for neural network modeling (including interaction terms flagged
   by PCA if desired).
4. Builds a neural network using Keras with elastic net regularization.
5. Uses early stopping and learning curve plots to monitor training.
6. Tunes hyperparameters (number of layers, units, dropout, regularization parameters).
7. Evaluates the final model using ROC, confusion matrix, classification metrics, and
   plots these diagnostics.
8. All plots and results are stored in the ./neural directory.

Usage: python neural_model.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# For PCA and correlation analysis
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# For neural network modeling
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping

# =============================================================================
# 0. Setup Directories
# =============================================================================
os.makedirs("./neural/plots", exist_ok=True)
os.makedirs("./neural/results", exist_ok=True)

# =============================================================================
# 1. Load Data and Standardize Feature Names
# =============================================================================
data_file = "./data/ncaa_2002_2024_cleaned.csv"
df = pd.read_csv(data_file)

# Standardize column names (replace spaces with underscores)
df.columns = df.columns.str.replace(" ", "_")

# We assume that the KenPom metrics have been merged into the dataset with names:
expected_metrics = [
    "NetRtg", "ORtg", "DRtg", "AdjT", "Luck", 
    "Strength_of_Schedule_NetRtg", "Strength_of_Schedule_ORtg", "Strength_of_Schedule_DRtg",
    "NCSOS_NetRtg"
]

# Create difference features (Team_A minus Team_B) for each metric
for metric in expected_metrics:
    colA = f"Team_A_{metric}"
    colB = f"Team_B_{metric}"
    diff_col = f"diff_{metric}"
    if colA in df.columns and colB in df.columns:
        df[diff_col] = df[colA] - df[colB]
    else:
        print(f"⚠️ Missing expected columns for {metric}")

# List of feature difference columns
feature_cols = [f"diff_{metric}" for metric in expected_metrics]

# Drop rows with missing values in feature_cols or in the target 'Winner'
df_model = df.dropna(subset=feature_cols + ["Winner"]).copy()
df_model["Winner"] = pd.to_numeric(df_model["Winner"], errors="coerce")
df_model = df_model.dropna(subset=["Winner"])

# =============================================================================
# 2. Correlation and PCA Analysis for Feature Reduction
# =============================================================================
# Compute correlation matrix for the difference features
corr_matrix = df_model[feature_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap of Feature Differences")
heatmap_file = "./neural/plots/correlation_heatmap.png"
plt.tight_layout()
plt.savefig(heatmap_file)
plt.close()
print(f"✅ Correlation heatmap saved to {heatmap_file}")

# Standardize features before PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_model[feature_cols])

# Perform PCA
pca = PCA()
pca.fit(X_scaled)
explained_var = pca.explained_variance_ratio_

plt.figure(figsize=(8, 6))
plt.plot(np.cumsum(explained_var), marker='o')
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA Explained Variance")
pca_plot_file = "./neural/plots/pca_explained_variance.png"
plt.tight_layout()
plt.savefig(pca_plot_file)
plt.close()
print(f"✅ PCA explained variance plot saved to {pca_plot_file}")

# Identify highly correlated features (e.g., correlation > 0.95) and drop the one with lower predictive power.
# We use the absolute correlation matrix and pick one from each pair.
threshold = 0.95
to_drop = set()
for i in range(len(feature_cols)):
    for j in range(i + 1, len(feature_cols)):
        f1, f2 = feature_cols[i], feature_cols[j]
        if abs(corr_matrix.loc[f1, f2]) > threshold:
            # Here we choose to drop the feature with lower absolute average correlation with the target
            corr_f1 = abs(df_model[f1].corr(df_model["Winner"]))
            corr_f2 = abs(df_model[f2].corr(df_model["Winner"]))
            if corr_f1 < corr_f2:
                to_drop.add(f1)
            else:
                to_drop.add(f2)

print("⚠️ Dropping the following highly correlated features:", to_drop)
df_model_reduced = df_model.drop(columns=list(to_drop))
# Update the feature columns list
feature_cols_reduced = [col for col in feature_cols if col not in to_drop]

with open("./neural/results/feature_reduction.txt", "w") as f:
    f.write("Original feature columns:\n" + ", ".join(feature_cols) + "\n\n")
    f.write("Dropped features due to high correlation (threshold > 0.95):\n" + ", ".join(to_drop) + "\n\n")
    f.write("Remaining features for modeling:\n" + ", ".join(feature_cols_reduced))
print("✅ Feature reduction results saved to ./neural/results/feature_reduction.txt")

# =============================================================================
# 3. Prepare Data for Neural Network Modeling
# =============================================================================
# Define X and y
X = df_model_reduced[feature_cols_reduced].values
y = df_model_reduced["Winner"].values

# Split into train and test sets (70/30 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# =============================================================================
# 4. Build a Neural Network Model with Elastic Net Regularization
# =============================================================================
# Here we define a function to build a Keras model.
# The regularization is applied using the l1_l2 kernel_regularizer.

def build_model(input_shape, hp):
    """
    Build a neural network model.
    hp: a dictionary of hyperparameters (or use Keras Tuner later for automatic tuning)
    """
    model = keras.Sequential()
    # Add an input layer with dropout
    model.add(layers.InputLayer(input_shape=(input_shape,)))
    model.add(layers.Dropout(hp.get("dropout_rate", 0.2)))
    
    # Add a hidden layer; number of neurons and activation can be tuned.
    model.add(layers.Dense(
        hp.get("units", 32),
        activation="relu",
        kernel_regularizer=regularizers.l1_l2(l1=hp.get("l1", 1e-4), l2=hp.get("l2", 1e-4))
    ))
    
    # Optionally add a second hidden layer if specified.
    if hp.get("use_second_layer", False):
        model.add(layers.Dense(
            hp.get("units2", 16),
            activation="relu",
            kernel_regularizer=regularizers.l1_l2(l1=hp.get("l1_2", 1e-4), l2=hp.get("l2_2", 1e-4))
        ))
    
    # Output layer: sigmoid activation for binary classification.
    model.add(layers.Dense(1, activation="sigmoid"))
    
    # Compile the model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=hp.get("lr", 1e-3)),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

# Hyperparameters dictionary (these values can be tuned)
hp = {
    "dropout_rate": 0.2,
    "units": 32,
    "l1": 1e-4,
    "l2": 1e-4,
    "use_second_layer": True,
    "units2": 16,
    "l1_2": 1e-4,
    "l2_2": 1e-4,
    "lr": 1e-3,
    "batch_size": 32,
    "epochs": 100
}

# Build the model
model = build_model(input_shape=X_train.shape[1], hp=hp)
model.summary(print_fn=lambda x: open("./neural/results/model_summary.txt", "a").write(x + "\n"))
print("✅ Neural network model summary appended to ./neural/results/model_summary.txt")

# =============================================================================
# 5. Train the Model with Early Stopping (to Prevent Overfitting)
# =============================================================================
early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    batch_size=hp["batch_size"],
    epochs=hp["epochs"],
    callbacks=[early_stop],
    verbose=1
)

# Save training history plots
plt.figure(figsize=(8, 6))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs. Validation Loss")
plt.legend()
loss_plot_file = "./neural/plots/loss_curve.png"
plt.tight_layout()
plt.savefig(loss_plot_file)
plt.close()
print(f"✅ Loss curve saved to {loss_plot_file}")

plt.figure(figsize=(8, 6))
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs. Validation Accuracy")
plt.legend()
acc_plot_file = "./neural/plots/accuracy_curve.png"
plt.tight_layout()
plt.savefig(acc_plot_file)
plt.close()
print(f"✅ Accuracy curve saved to {acc_plot_file}")

# =============================================================================
# 6. Evaluate the Model on the Test Set
# =============================================================================
y_pred_proba = model.predict(X_test).ravel()
y_pred = (y_pred_proba >= 0.5).astype(int)
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

print(f"✅ Test Loss: {test_loss:.3f}, Test Accuracy: {test_acc:.3f}")

# Save evaluation metrics to a file
report = classification_report(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
with open("./neural/results/evaluation_report.txt", "w") as f:
    f.write(f"Test Loss: {test_loss:.3f}\nTest Accuracy: {test_acc:.3f}\n\n")
    f.write("Confusion Matrix:\n" + np.array2string(cm) + "\n\n")
    f.write("Classification Report:\n" + report)
print("✅ Evaluation report saved to ./neural/results/evaluation_report.txt")

# ROC Curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for Neural Network Model")
plt.legend(loc="lower right")
roc_plot_file = "./neural/plots/nn_roc_curve.png"
plt.tight_layout()
plt.savefig(roc_plot_file)
plt.close()
print(f"✅ ROC curve saved to {roc_plot_file}")

# =============================================================================
# 7. Additional Evaluation: Cross-Validation and Learning Curve (Optional)
# =============================================================================
# Here we suggest additional methods without implementing full k-fold CV in this script.
# You can perform k-fold CV (e.g., using Keras wrappers and scikit-learn’s cross_val_score) 
# and plot learning curves (training vs. validation metrics vs. training set size) to ensure robustness.

# For instance, one might:
# - Use sklearn.model_selection.KFold to split data into k folds and record the validation loss/accuracy.
# - Plot the learning curve by training the model on increasing subsets of the training data.
#
# These approaches help diagnose overfitting and underfitting and assess the stability of the network.

print("✅ Neural network training and diagnostics complete. All plots and results are saved in the ./neural folder.")
