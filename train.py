
import os
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Create required folders
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# 1. Load dataset
df = pd.read_csv(
    r"data/month_3_ann_student_performance_dataset_15000.csv"
)

print("Dataset Shape:", df.shape)
print("\nFirst Five Records:")
print(df.head())

print("\nDataset Information:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Records:", df.duplicated().sum())

print("\nClass Distribution:")
print(df["performance_category"].value_counts())

# 2. Remove duplicate records
df = df.drop_duplicates().copy()

# 3. Separate features and target
FEATURES = [
    "age",
    "attendance_pct",
    "study_hours_per_week",
    "previous_score",
    "assignment_completion_pct",
    "internal_assessment_score",
    "participation_score",
    "sleep_hours_per_day",
    "extracurricular_hours_per_week",
    "failed_subjects"
]

TARGET = "performance_category"

# Handle missing values if any are present
for col in FEATURES:
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df[col] = df[col].fillna(df[col].median())

df = df.dropna(subset=[TARGET])

X = df[FEATURES]
y = df[TARGET].astype(str)

# 4. Encode target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print("\nEncoded Classes:")
print(label_encoder.classes_)

# 5. Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)

# 6. Feature scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save preprocessing objects
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(label_encoder, "models/label_encoder.pkl")

print("\nTraining Data:", X_train_scaled.shape)
print("Testing Data:", X_test_scaled.shape)

# 7. Baseline Model: Logistic Regression
baseline = LogisticRegression(max_iter=1000)

baseline.fit(X_train_scaled, y_train)

baseline_predictions = baseline.predict(X_test_scaled)

baseline_accuracy = accuracy_score(
    y_test, baseline_predictions
)

print("\nLogistic Regression Accuracy:",
      round(baseline_accuracy * 100, 2), "%")

# 8. Build Artificial Neural Network
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(len(FEATURES),)),

    tf.keras.layers.Dense(64, activation="relu"),

    tf.keras.layers.Dense(32, activation="relu"),

    tf.keras.layers.Dense(
        len(label_encoder.classes_),
        activation="softmax"
    )
])

# 9. Compile ANN
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# 10. Early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

# 11. Train ANN
history = model.fit(
    X_train_scaled,
    y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.20,
    callbacks=[early_stopping],
    verbose=1
)

# 12. Evaluate ANN
test_loss, test_accuracy = model.evaluate(
    X_test_scaled,
    y_test,
    verbose=0
)

print("\nANN Test Accuracy:",
      round(test_accuracy * 100, 2), "%")

# 13. Make predictions
probabilities = model.predict(X_test_scaled)

predictions = np.argmax(probabilities, axis=1)

# 14. Classification Report
report = classification_report(
    y_test,
    predictions,
    labels=np.arange(len(label_encoder.classes_)),
    target_names=label_encoder.classes_,
    zero_division=0
)

print("\nANN Classification Report:")
print(report)

with open("results/classification_report.txt", "w") as f:
    f.write("Logistic Regression Accuracy: ")
    f.write(str(baseline_accuracy))
    f.write("\nANN Test Accuracy: ")
    f.write(str(test_accuracy))
    f.write("\n\nANN Classification Report:\n")
    f.write(report)

# 15. Confusion Matrix
cm = confusion_matrix(
    y_test,
    predictions,
    labels=np.arange(len(label_encoder.classes_))
)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_
)

plt.xlabel("Predicted Category")
plt.ylabel("Actual Category")
plt.title("ANN Confusion Matrix")

plt.tight_layout()
plt.savefig("results/confusion_matrix.png")
plt.show()

# 16. Accuracy and Loss Graphs
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Training")
plt.plot(history.history["val_accuracy"], label="Validation")
plt.title("ANN Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Training")
plt.plot(history.history["val_loss"], label="Validation")
plt.title("ANN Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.savefig("results/training_graphs.png")
plt.show()

# 17. Save trained ANN
model.save("models/student_performance.keras")

print("\nModel saved successfully!")
print("Training completed!")