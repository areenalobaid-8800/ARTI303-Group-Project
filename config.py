from pathlib import Path

# Project directory
PROJECT_DIR = Path(__file__).resolve().parent

# Data directories and files
RAW_DATA_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_DIR / "data" / "processed"
RAW_DATA_FILE = RAW_DATA_DIR / "StudentPerformanceFactors.csv"

# Other directories
VISUALIZATIONS_DIR = PROJECT_DIR / "visualizations"
MODELS_DIR = PROJECT_DIR / "models"
LOGS_DIR = PROJECT_DIR / "logs"

# AI settings
TARGET_COLUMN = "Exam_Score"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Valid exam score range
SCORE_MIN = 0
SCORE_MAX = 100
# Expected numeric columns and their allowed ranges
NUMERIC_RANGES = {
    "Hours_Studied": (0, 168),
    "Attendance": (0, 100),
    "Sleep_Hours": (0, 24),
    "Previous_Scores": (0, 100),
    "Tutoring_Sessions": (0, 50),
    "Physical_Activity": (0, 168),
}

# Expected categorical columns and their allowed values
CATEGORICAL_VALUES = {
    "Parental_Involvement": ["Low", "Medium", "High"],
    "Access_to_Resources": ["Low", "Medium", "High"],
    "Extracurricular_Activities": ["No", "Yes"],
    "Motivation_Level": ["Low", "Medium", "High"],
    "Internet_Access": ["No", "Yes"],
    "Family_Income": ["Low", "Medium", "High"],
    "Teacher_Quality": ["Low", "Medium", "High"],
    "School_Type": ["Public", "Private"],
    "Peer_Influence": ["Negative", "Neutral", "Positive"],
    "Learning_Disabilities": ["No", "Yes"],
    "Parental_Education_Level": ["High School", "College", "Postgraduate"],
    "Distance_from_Home": ["Near", "Moderate", "Far"],
    "Gender": ["Male", "Female"],
}