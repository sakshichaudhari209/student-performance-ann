
import streamlit as st
from predictor import predict_performance

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 AI-Based Student Performance Risk Predictor")

st.write(
    "Enter student academic and behavioral information "
    "to predict the expected performance category."
)

st.divider()

st.subheader("Student Information")

with st.form("student_form"):

    age = st.number_input(
        "Age",
        min_value=17,
        max_value=24,
        value=20
    )

    attendance = st.slider(
        "Attendance (%)",
        45.0, 100.0, 82.0
    )

    study_hours = st.slider(
        "Study Hours Per Week",
        2.0, 40.0, 15.0
    )

    previous_score = st.slider(
        "Previous Score",
        25.0, 100.0, 71.0
    )

    assignment = st.slider(
        "Assignment Completion (%)",
        20.0, 100.0, 80.0
    )

    internal_score = st.slider(
        "Internal Assessment Score",
        20.0, 100.0, 68.0
    )

    participation = st.slider(
        "Participation Score",
        1, 10, 7
    )

    sleep = st.slider(
        "Sleep Hours Per Day",
        4.0, 10.0, 7.0
    )

    extracurricular = st.slider(
        "Extracurricular Hours Per Week",
        0.0, 18.0, 3.0
    )

    failed_subjects = st.slider(
        "Number of Failed Subjects",
        0, 5, 0
    )

    submitted = st.form_submit_button(
        "Predict Performance"
    )

if submitted:

    student_data = {
        "age": age,
        "attendance_pct": attendance,
        "study_hours_per_week": study_hours,
        "previous_score": previous_score,
        "assignment_completion_pct": assignment,
        "internal_assessment_score": internal_score,
        "participation_score": participation,
        "sleep_hours_per_day": sleep,
        "extracurricular_hours_per_week": extracurricular,
        "failed_subjects": failed_subjects
    }

    result = predict_performance(student_data)

    category = result["predicted_category"]

    st.divider()

    st.subheader("Prediction Result")

    if category == "High":
        st.success("Predicted Performance: HIGH")

    elif category == "Medium":
        st.warning("Predicted Performance: MEDIUM")

    else:
        st.error("Predicted Performance: LOW")

    st.metric(
        "Prediction Confidence",
        f"{result['confidence']}%"
    )

    st.subheader("Class Probabilities")

    for label, probability in result["probabilities"].items():

        st.write(f"{label}: {probability}%")

        st.progress(
            int(round(probability))
        )

    st.info(
        "This result is an ML-based estimate, "
        "not a guaranteed academic outcome."
    )