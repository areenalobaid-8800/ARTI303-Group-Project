
"""Streamlit Application: Student Performance Prediction (SPP).

Predicts a student's exam score using Regression and detects
unusual student data patterns using Anomaly Detection.
"""

import logging

import pandas as pd
import streamlit as st

import config
from AI_Prediction import StudentPerformancePredictor


# --------------------------------------------------
# 1. Application Configuration
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide",
)


NUMERIC_DEFAULTS = {
    "Hours_Studied": 10,
    "Attendance": 90,
    "Sleep_Hours": 8,
    "Previous_Scores": 70,
    "Tutoring_Sessions": 1,
    "Physical_Activity": 3,
}

CATEGORICAL_DEFAULTS = {
    "Parental_Involvement": "Medium",
    "Access_to_Resources": "Medium",
    "Extracurricular_Activities": "Yes",
    "Motivation_Level": "Medium",
    "Internet_Access": "Yes",
    "Family_Income": "Medium",
    "Teacher_Quality": "Medium",
    "School_Type": "Public",
    "Peer_Influence": "Positive",
    "Learning_Disabilities": "No",
    "Parental_Education_Level": "College",
    "Distance_from_Home": "Near",
    "Gender": "Female",
}


# --------------------------------------------------
# 2. Load Prediction System
# --------------------------------------------------

@st.cache_resource
def load_predictor():
    """Load and cache the prediction system."""

    predictor = StudentPerformancePredictor()

    logger.info(
        "Prediction system loaded successfully."
    )

    return predictor


# --------------------------------------------------
# 3. Student Input Form
# --------------------------------------------------

def student_input_form():
    """Display the student information form."""

    input_data = {}

    st.subheader("Student Information")

    st.write(
        "Enter the student's academic and background information "
        "to predict the exam score and detect unusual data patterns."
    )

    with st.form("student_input_form"):

        st.markdown("### Academic and Numeric Information")

        numeric_columns = st.columns(3)

        for index, (column, limits) in enumerate(
            config.NUMERIC_RANGES.items()
        ):
            minimum, maximum = limits
            default = NUMERIC_DEFAULTS[column]

            step = 0.5 if column == "Sleep_Hours" else 1.0

            number_format = (
                "%.1f" if column == "Sleep_Hours" else "%.0f"
            )

            with numeric_columns[index % 3]:

                input_data[column] = st.number_input(
                    label=column.replace("_", " "),
                    min_value=float(minimum),
                    max_value=float(maximum),
                    value=float(default),
                    step=step,
                    format=number_format,
                    key=f"input_{column}",
                )

        st.divider()

        st.markdown(
            "### Student Background and Learning Information"
        )

        categorical_columns = st.columns(2)

        for index, (column, allowed_values) in enumerate(
            config.CATEGORICAL_VALUES.items()
        ):
            default = CATEGORICAL_DEFAULTS[column]

            with categorical_columns[index % 2]:

                input_data[column] = st.selectbox(
                    label=column.replace("_", " "),
                    options=allowed_values,
                    index=allowed_values.index(default),
                    key=f"input_{column}",
                )

        st.divider()

        submitted = st.form_submit_button(
            "Analyze Student Performance",
            type="primary",
            use_container_width=True,
        )

    return input_data, submitted


# --------------------------------------------------
# 4. Analyze Student
# --------------------------------------------------

def analyze_student(input_data, predictor):
    """Generate predictions using the AI prediction module."""

    results = predictor.predict(input_data)

    logger.info(
        "Student analysis completed: score=%.2f, status=%s",
        results["predicted_exam_score"],
        results["anomaly_status"],
    )

    return results


# --------------------------------------------------
# 5. Display Prediction Results
# --------------------------------------------------

def display_results(result, input_data):
    """Display regression and anomaly detection results."""

    st.divider()
    st.header("Student Analysis Results")

    st.write(
        "The AI models have analyzed the student's information."
    )

    # Regression result
    st.subheader("Exam Score Prediction")

    predicted_score = result["predicted_exam_score"]

    st.metric(
        label="Predicted Exam Score",
        value=f"{predicted_score:.2f} / 100",
    )

    # Keep the original prediction unchanged.
    # Limit only the visual progress indicator.
    progress_value = max(
        0.0,
        min(1.0, predicted_score / 100),
    )

    st.progress(progress_value)

    if not config.SCORE_MIN <= predicted_score <= config.SCORE_MAX:
        st.warning(
            "The predicted score is outside the configured "
            "exam score range. Please interpret this result "
            "with caution."
        )

    # Anomaly detection result
    st.divider()
    st.subheader("Anomaly Detection Result")

    if result["anomaly_status"] == "Anomaly":

        st.warning(
            "Unusual Data Pattern Detected",
            icon="⚠️",
        )

        st.write(
            "The student's input contains a pattern that differs "
            "from the patterns learned by the model. "
            "The teacher may review the information."
        )

    else:

        st.success(
            "No Unusual Data Pattern Detected",
            icon="✅",
        )

        st.write(
            "The student's input follows patterns that are "
            "generally consistent with the training data."
        )

    st.metric(
        label="Anomaly Score",
        value=f"{result['anomaly_score']:.4f}",
    )

    st.caption(
        "Lower anomaly scores indicate more unusual observations. "
        "The score is a model output, not a probability."
    )

    # Numeric input visualization
    st.divider()
    st.subheader("Student Information Visualization")

    normalized_values = {}

    for column, (minimum, maximum) in config.NUMERIC_RANGES.items():

        value = input_data[column]

        if maximum > minimum:

            normalized_values[column.replace("_", " ")] = (
                (value - minimum) / (maximum - minimum)
            ) * 100

        else:

            normalized_values[column.replace("_", " ")] = 0

    chart_df = pd.DataFrame(
        {
            "Feature": list(normalized_values.keys()),
            "Value (% of allowed range)": list(
                normalized_values.values()
            ),
        }
    ).set_index("Feature")

    st.bar_chart(chart_df)

    st.caption(
        "The chart shows each numeric input as a percentage "
        "of its configured allowed range. It does not represent "
        "the student's academic performance."
    )

    # Important note
    st.divider()

    st.subheader("Important Note")

    st.info(
        "An anomaly does not necessarily indicate poor academic "
        "performance or incorrect information. It identifies an "
        "unusual pattern that may require teacher review."
    )


# --------------------------------------------------
# 6. Main Application
# --------------------------------------------------

def main():
    """Run the Student Performance Prediction application."""

    st.title("🎓 Student Performance Prediction")

    st.markdown(
        """
        Welcome to the **Student Performance Prediction (SPP)** system.

        This application helps teachers analyze student information
        using Artificial Intelligence.

        It provides two AI-based functions:

        - **Exam Score Prediction:** Estimates the student's expected exam score.
        - **Anomaly Detection:** Identifies unusual patterns in student information
          for teacher review.
        """
    )

    st.divider()

    # Load trained models and preprocessing pipeline.
    try:

        predictor = load_predictor()

    except FileNotFoundError as exc:

        logger.error("Required model file is missing: %s", exc)

        st.error(str(exc))

        st.info(
            "Make sure the trained Regression model, "
            "Anomaly Detection model, and preprocessor "
            "are saved inside the project's models folder."
        )

        return

    except Exception as exc:

        logger.exception("Failed to load the prediction system.")

        st.error(
            f"An error occurred while loading the AI models: {exc}"
        )

        return

    # Collect student information.
    input_data, submitted = student_input_form()

    if submitted:

        try:

            with st.spinner(
                "Analyzing student information..."
            ):

                result = analyze_student(
                    input_data,
                    predictor,
                )

            st.success(
                "Student analysis completed successfully!"
            )

            display_results(
                result,
                input_data,
            )

        except (ValueError, TypeError, KeyError) as exc:

            logger.warning(
                "Student input or analysis error: %s",
                exc,
            )

            st.error(
                f"Invalid input or analysis error: {exc}"
            )

        except Exception:

            logger.exception(
                "Unexpected error during student analysis."
            )

            st.error(
                "An unexpected error occurred while analyzing "
                "the student's information."
            )

    st.divider()

    st.caption(
        "Student Performance Prediction (SPP) | "
        "AI-Based Academic Performance Analysis"
    )


if __name__ == "__main__":
    main()
