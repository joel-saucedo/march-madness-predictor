"""
hyperparameter_tuning.py

This script loads the cleaned NCAA tournament and KenPom dataset, uses prior PCA work to drop redundant features,
and then employs hyperparameter tuning via Keras Tuner's Hyperband to search for the optimal neural network configuration.
The objective is to maximize the ROC AUC on a validation set while applying elastic net (L1+L2) regularization.

All results (best hyperparameters, diagnostic plots, etc.) are saved in the ./neural folder.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping

# Import Keras Tuner (make sure to install it: pip install keras-tuner)
import kerastuner as kt

# =============================================================================
# 0. Setup Directories
# =============================================================================
os.makedirs("./neural/plots", exist_ok=True)
os.makedirs("./neural/results", exist_ok=True)

# =============================================================================
# 1. Load Data and Preprocess Features
# =============================================================================
data_file = "./data/ncaa_2002_2024_cleaned.csv"
df = pd.read_csv(data_file)

# Standardize column names (replace spaces with underscores)
df.columns = df.columns.str.replace(" ", "_")

# Define expected KenPom metrics and create difference features (Team_A - Team_B)
expected_metrics = [
    "NetRtg", "ORtg", "DRtg", "AdjT", "Luck", 
    "Strength_of_Schedule_NetRtg", "Strength_of_Schedule_ORtg", "Strength_of_Schedule_DRtg",
    "NCSOS_NetRtg"
]

for metric in expected_metrics:
    colA = f"Team_A_{metric}"
    colB = f"Team_B_{metric}"
    diff_col = f"diff_{metric}"
    if colA in df.columns and colB in df.columns:
        df[diff_col] = df[colA] - df[colB]
    else:
        print(f"⚠️ Missing expected columns for {metric}")

# All difference features
feature_cols = [f"diff_{metric}" for metric in expected_metrics]

# Based on PCA results, drop two highly correlated features:
drop_features = ["diff_Strength_of_Schedule_ORtg", "diff_Strength_of_Schedule_DRtg"]
reduced_features = [col for col in feature_cols if col not in drop_features]

# Drop rows with missing values in our features or target "Winner"
df_model = df.dropna(subset=reduced_features + ["Winner"]).copy()
df_model["Winner"] = pd.to_numeric(df_model["Winner"], errors="coerce")
df_model = df_model.dropna(subset=["Winner"])

# =============================================================================
# 2. Prepare Data for Modeling
# =============================================================================
X = df_model[reduced_features].values
y = df_model["Winner"].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# =============================================================================
# 3. Define Model Builder for Hyperparameter Tuning
# =============================================================================
def build_model(hp):
    """
    Build a neural network model with hyperparameters to be tuned.
    Hyperparameters include dropout rate, number of units, regularization parameters, learning rate,
    and whether to use a second hidden layer.
    """
    model = keras.Sequential()
    model.add(layers.InputLayer(input_shape=(X_train.shape[1],)))
    
    # Dropout rate
    dropout_rate = hp.Float("dropout_rate", 0.1, 0.5, step=0.1, default=0.2)
    model.add(layers.Dropout(dropout_rate))
    
    # First hidden layer
    units = hp.Int("units", min_value=16, max_value=64, step=16, default=32)
    l1_reg = hp.Float("l1_reg", 1e-5, 1e-2, sampling="LOG", default=1e-4)
    l2_reg = hp.Float("l2_reg", 1e-5, 1e-2, sampling="LOG", default=1e-4)
    model.add(layers.Dense(units, activation="relu",
                           kernel_regularizer=regularizers.l1_l2(l1=l1_reg, l2=l2_reg)))
    
    # Optionally add a second hidden layer
    if hp.Boolean("use_second_layer", default=True):
        units2 = hp.Int("units2", min_value=8, max_value=32, step=8, default=16)
        l1_reg2 = hp.Float("l1_reg2", 1e-5, 1e-2, sampling="LOG", default=1e-4)
        l2_reg2 = hp.Float("l2_reg2", 1e-5, 1e-2, sampling="LOG", default=1e-4)
        model.add(layers.Dense(units2, activation="relu",
                               kernel_regularizer=regularizers.l1_l2(l1=l1_reg2, l2=l2_reg2)))
    
    # Output layer
    model.add(layers.Dense(1, activation="sigmoid"))
    
    # Learning rate
    lr = hp.Float("learning_rate", 1e-4, 1e-2, sampling="LOG", default=1e-3)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                  loss="binary_crossentropy",
                  metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])
    return model

# =============================================================================
# 4. Hyperparameter Tuning with Keras Tuner Hyperband
# =============================================================================
tuner = kt.Hyperband(
    build_model,
    objective=kt.Objective("val_auc", direction="max"),
    max_epochs=50,
    factor=3,
    directory="./neural/results",
    project_name="hyperparameter_tuning"
)

stop_early = EarlyStopping(monitor="val_loss", patience=5)
tuner.search(X_train, y_train, epochs=50, validation_split=0.2, callbacks=[stop_early])

# Get best hyperparameters and save them
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
with open("./neural/results/best_hyperparameters.txt", "w") as f:
    f.write("Best Hyperparameters:\n")
    for key, value in best_hps.values.items():
        f.write(f"{key}: {value}\n")
print("✅ Best hyperparameters saved to ./neural/results/best_hyperparameters.txt")

# =============================================================================
# 5. Retrain Model with Best Hyperparameters
# =============================================================================
model = tuner.hypermodel.build(best_hps)
history = model.fit(
    X_train, y_train,
    epochs=50,
    validation_split=0.2,
    callbacks=[EarlyStopping(monitor="val_loss", patience=5)],
    verbose=1
)

# =============================================================================
# 6. Evaluate the Tuned Model
# =============================================================================
test_loss, test_acc, test_auc = model.evaluate(X_test, y_test, verbose=0)
print(f"✅ Tuned Model - Test Loss: {test_loss:.3f}, Test Accuracy: {test_acc:.3f}, Test AUC: {test_auc:.3f}")

# Save evaluation results
with open("./neural/results/evaluation_report.txt", "w") as f:
    f.write(f"Test Loss: {test_loss:.3f}\n")
    f.write(f"Test Accuracy: {test_acc:.3f}\n")
    f.write(f"Test AUC: {test_auc:.3f}\n")
print("✅ Evaluation report saved to ./neural/results/evaluation_report.txt")

# =============================================================================
# 7. Plot Training History and ROC Curve
# =============================================================================
# Loss curve
plt.figure(figsize=(8, 6))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Tuned Model: Training vs. Validation Loss")
plt.legend()
plt.tight_layout()
plt.savefig("./neural/plots/tuned_loss_curve.png")
plt.close()

# AUC curve
plt.figure(figsize=(8, 6))
plt.plot(history.history["auc"], label="Training AUC")
plt.plot(history.history["val_auc"], label="Validation AUC")
plt.xlabel("Epoch")
plt.ylabel("AUC")
plt.title("Tuned Model: Training vs. Validation AUC")
plt.legend()
plt.tight_layout()
plt.savefig("./neural/plots/tuned_auc_curve.png")
plt.close()

# ROC curve on test set
from sklearn.metrics import roc_curve, auc
y_pred_proba = model.predict(X_test).ravel()
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"Tuned ROC Curve (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for Tuned Neural Network Model")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("./neural/plots/tuned_roc_curve.png")
plt.close()

print("✅ Hyperparameter tuning and evaluation complete. All plots and results are saved in the ./neural folder.")
