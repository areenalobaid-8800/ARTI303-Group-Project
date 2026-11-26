

import pandas as pd
import joblib

# 1. Load the saved models and preprocessing tool
preprocessor = joblib.load("models/preprocessor.pkl")
regression_model = joblib.load("models/regression_model.pkl")
classification_model = joblib.load("models/classification_model.pkl")


# 2. Enter new student data
new_student = pd.DataFrame([{
    "Hours_Studied": float(input("Hours Studied: ")),
    "Attendance": float(input("Attendance: ")),
    "Sleep_Hours": float(input("Sleep Hours: ")),
    "Previous_Scores": float(input("Previous Scores: ")),
    "Tutoring_Sessions": float(input("Tutoring Sessions: ")),
    "Physical_Activity": float(input("Physical Activity: ")),

    "Parental_Involvement": input("Parental Involvement: "),
    "Access_to_Resources": input("Access to Resources: "),
    "Extracurricular_Activities": input("Extracurricular Activities: "),
    "Motivation_Level": input("Motivation Level: "),
    "Internet_Access": input("Internet Access: "),
    "Family_Income": input("Family Income: "),
    "Teacher_Quality": input("Teacher Quality: "),
    "School_Type": input("School Type: "),
    "Peer_Influence": input("Peer Influence: "),
    "Learning_Disabilities": input("Learning Disabilities: "),
    "Parental_Education_Level": input("Parental Education Level: "),
    "Distance_from_Home": input("Distance from Home: "),
    "Gender": input("Gender: ")
}])


# 3. Transform the new student data
student_transformed = preprocessor.transform(new_student)


# 4. Predict the exam score
predicted_score = regression_model.predict(student_transformed)


# 5. Predict performance level
predicted_class = classification_model.predict(student_transformed)


# 6. Display the results
print("\nStudent Performance Prediction")
print("--------------------------------")
print("Predicted Exam Score:", round(predicted_score[0], 2))
print("Predicted Performance Level:", predicted_class[0])

