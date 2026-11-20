
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from predictor import predict_performance

app = FastAPI(
    title="Student Performance Prediction API",
    description="ANN-based student performance prediction",
    version="1.0"
)

# Define input structure
class StudentData(BaseModel):

    age: int = Field(ge=17, le=24)

    attendance_pct: float = Field(ge=45, le=100)

    study_hours_per_week: float = Field(ge=2, le=40)

    previous_score: float = Field(ge=25, le=100)

    assignment_completion_pct: float = Field(ge=20, le=100)

    internal_assessment_score: float = Field(ge=20, le=100)

    participation_score: int = Field(ge=1, le=10)

    sleep_hours_per_day: float = Field(ge=4, le=10)

    extracurricular_hours_per_week: float = Field(ge=0, le=18)

    failed_subjects: int = Field(ge=0, le=5)

@app.get("/")
def home():

    return {
        "message": "Student Performance Prediction API",
        "status": "Running"
    }

@app.post("/predict")
def predict(student: StudentData):

    try:

        student_data = student.model_dump()

        result = predict_performance(student_data)

        return {
            "success": True,
            "prediction": result["predicted_category"],
            "confidence": result["confidence"],
            "probabilities": result["probabilities"]
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )