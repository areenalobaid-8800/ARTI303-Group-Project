
from pathlib import Path

# Project directory
PROJECT_DIR = Path(__file__).resolve().parent

# Data directories
RAW_DATA_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_DIR / "data" / "processed"

# Visualization directory
VISUALIZATIONS_DIR = PROJECT_DIR / "visualizations"

# AI settings
TARGET_COLUMN = "Exam_Score"
TEST_SIZE = 0.2
RANDOM_STATE = 42
