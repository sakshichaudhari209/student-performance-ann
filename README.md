# student-performance-ann
# AI-Based Student Performance Risk Predictor

## 1. Project overview
This beginner-to-intermediate project trains a feed-forward Artificial Neural Network
(ANN) to classify a student's expected performance into the categories contained in
the assigned dataset: Low, Medium, and High.

The project also includes:
- Logistic Regression baseline
- ANN evaluation on a held-out test set
- Saved Keras model and preprocessing objects
- Streamlit prediction application
- FastAPI JSON prediction endpoint
- Jupyter Notebook for exploration and training

## 2. Dataset
Place the assigned CSV at `data/student_data.csv`.

Target column:
- `performance_category`

Input features:
1. `age`
2. `attendance_pct`
3. `study_hours_per_week`
4. `previous_score`
5. `assignment_completion_pct`
6. `internal_assessment_score`
7. `participation_score`
8. `sleep_hours_per_day`
9. `extracurricular_hours_per_week`
10. `failed_subjects`

`student_id` is an identifier and is not used as a model feature.

## 3. ANN architecture
- Input: 10 features
- Dense hidden layer: 64 neurons, ReLU
- Dense hidden layer: 32 neurons, ReLU
- Output: 3 neurons, Softmax
- Optimizer: Adam
- Loss: Sparse categorical crossentropy
- Metric: Accuracy
- Early stopping: monitors validation loss

The actual number of output classes is taken from the target labels during training.

## 4. Environment setup
Recommended: Python 3.11 in a virtual environment. TensorFlow support depends
on the Python version and operating system.

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation for this terminal:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## 5. Train the model

```powershell
python train.py
```

Training creates:
- `models/student_performance.keras`
- `models/scaler.joblib`
- `models/label_encoder.joblib`
- `models/feature_medians.joblib`
- `models/logistic_regression.joblib`
- `models/metadata.json`
- `results/classification_report.txt`
- `results/metrics.json`
- `results/confusion_matrix.png`
- `results/training_curves.png`

The scaler and imputation values are fitted using training data only. The test
set is held out for final evaluation.

## 6. Run the Streamlit application

```powershell
streamlit run app.py
```

Usually opens at http://localhost:8501.

## 7. Run the API

Open another PowerShell terminal, activate the same virtual environment, then run:

```powershell
uvicorn api:app --reload
```

API home: http://127.0.0.1:8000  
Interactive documentation: http://127.0.0.1:8000/docs

Use POST `/predict` with JSON containing all ten input features.

## 8. Notebook
Open Jupyter:

```powershell
jupyter notebook
```

Open `notebooks/ann_training.ipynb` and run the cells from top to bottom.
The notebook contains dataset inspection, visualizations, preprocessing,
baseline model, ANN training, evaluation, and artifact saving.

## 9. Evaluation
Report the measured test accuracy, macro precision, macro recall, macro F1-score,
and confusion matrix. Compare Logistic Regression and ANN using the same test set.
Do not claim that ANN is better unless the measured results support that conclusion.

## 10. Responsible use
Predictions are estimates based on patterns in the training dataset. They are not
guarantees of future academic outcomes and should not be used as the sole basis
for consequential decisions about students. Review model errors and possible bias.
