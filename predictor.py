
import numpy as np
import joblib
import tensorflow as tf

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

# Load saved model and preprocessing objects
model = tf.keras.models.load_model(
    "models/student_performance.keras"
)

scaler = joblib.load("models/scaler.pkl")

label_encoder = joblib.load(
    "models/label_encoder.pkl"
)

def predict_performance(student_data):

    # Arrange features in training order
    values = np.array([
        student_data[feature]
        for feature in FEATURES
    ]).reshape(1, -1)

    # Apply the same scaling used during training
    scaled_values = scaler.transform(values)

    # Get probabilities
    probabilities = model.predict(
        scaled_values,
        verbose=0
    )[0]

    # Find predicted class
    predicted_index = int(np.argmax(probabilities))

    predicted_category = label_encoder.inverse_transform(
        [predicted_index]
    )[0]

    # Return prediction and probabilities
    class_probabilities = {
        label: round(float(probability) * 100, 2)
        for label, probability in zip(
            label_encoder.classes_,
            probabilities
        )
    }

    return {
        "predicted_category": predicted_category,
        "confidence": round(
            float(probabilities[predicted_index]) * 100, 2
        ),
        "probabilities": class_probabilities
    }
