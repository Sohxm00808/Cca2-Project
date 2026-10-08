# Name: Jaylaksh Kawathekar
# PRN : 1302250524
# -----------------------------------------------------------------------------
# Traditional Programming vs Machine Learning Programming
# Problem: Predict whether a student will Pass or Fail.
# Input  : input_data.xlsx (sheet "Data", header on row 4)
#
# Traditional programming : a human writes the RULES, the program applies them.
#                           (Data + Rules  -> Answers)
# Machine learning        : the program LEARNS the rules from labelled examples.
#                           (Data + Answers -> Rules/Model)
# -----------------------------------------------------------------------------

import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, confusion_matrix

INPUT_FILE = "input_data.xlsx"
SHEET_NAME = "Data"
HEADER_ROW = 3          # row 4 in Excel (0-indexed for pandas)
FEATURES = ["Hours_Studied", "Attendance_Pct", "Assignments_Submitted", "Previous_Score"]
TARGET = "Result"


def load_data(path=INPUT_FILE):
    """Read the dataset from the Excel file."""
    try:
        df = pd.read_excel(path, sheet_name=SHEET_NAME, header=HEADER_ROW)
    except FileNotFoundError:
        sys.exit(f"Input file '{path}' not found. Keep it in the same folder as this script.")
    df = df.dropna(subset=FEATURES + [TARGET])
    return df


# ------------------------------ TRADITIONAL ----------------------------------
def traditional_predict(row):
    """Hand-written rules decided by the programmer."""
    if row["Attendance_Pct"] < 50:
        return "Fail"
    if row["Previous_Score"] >= 60 and row["Attendance_Pct"] >= 70:
        return "Pass"
    if row["Hours_Studied"] >= 4 and row["Assignments_Submitted"] >= 6:
        return "Pass"
    if row["Previous_Score"] >= 45 and row["Hours_Studied"] >= 3 and row["Attendance_Pct"] >= 60:
        return "Pass"
    return "Fail"


def run_traditional(df):
    preds = df.apply(traditional_predict, axis=1)
    acc = accuracy_score(df[TARGET], preds)
    return preds, acc


# --------------------------- MACHINE LEARNING --------------------------------
def run_ml(df):
    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    model = DecisionTreeClassifier(max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    return model, X_test, y_test, preds, acc


def main():
    print("Name: Jaylaksh Kawathekar | PRN: 1302250524")
    print("=" * 70)

    df = load_data()
    print(f"Loaded {len(df)} records from '{INPUT_FILE}'")
    print(df.head().to_string(index=False))

    # 1) Traditional programming
    print("\n--- 1) TRADITIONAL PROGRAMMING (hand-written rules) ---")
    trad_preds, trad_acc = run_traditional(df)
    print(f"Accuracy on all {len(df)} records: {trad_acc:.2%}")

    # 2) Machine learning
    print("\n--- 2) MACHINE LEARNING (Decision Tree learned from data) ---")
    model, X_test, y_test, ml_preds, ml_acc = run_ml(df)
    print(f"Accuracy on unseen test data ({len(y_test)} records): {ml_acc:.2%}")
    print("Confusion matrix [Fail, Pass]:")
    print(confusion_matrix(y_test, ml_preds, labels=["Fail", "Pass"]))
    print("\nRules the model learned by itself:")
    print(export_text(model, feature_names=FEATURES))

    # Fair comparison: traditional rules on the same test split
    trad_test_preds = df.loc[X_test.index].apply(traditional_predict, axis=1)
    trad_test_acc = accuracy_score(y_test, trad_test_preds)

    print("--- COMPARISON (same test records) ---")
    print(f"Traditional rules : {trad_test_acc:.2%}")
    print(f"Machine learning  : {ml_acc:.2%}")

    # 3) Predict for new students
    print("\n--- PREDICTING NEW STUDENTS ---")
    new_students = pd.DataFrame(
        [
            {"Hours_Studied": 6.0, "Attendance_Pct": 90, "Assignments_Submitted": 9, "Previous_Score": 78},
            {"Hours_Studied": 1.0, "Attendance_Pct": 45, "Assignments_Submitted": 2, "Previous_Score": 35},
            {"Hours_Studied": 3.5, "Attendance_Pct": 72, "Assignments_Submitted": 5, "Previous_Score": 55},
        ]
    )
    new_students["Traditional"] = new_students.apply(traditional_predict, axis=1)
    new_students["ML"] = model.predict(new_students[FEATURES])
    print(new_students.to_string(index=False))

    print("\nKey idea: traditional code needs a programmer to know the rules;")
    print("ML code discovers the rules from data and can improve with more data.")


if __name__ == "__main__":
    main()
