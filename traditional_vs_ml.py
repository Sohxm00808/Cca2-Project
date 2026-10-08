"""
Name : Soham Ashok Jagtap
PRN  : 1302250127
Topic: Traditional Programming vs Machine Learning (Loan Approval)

Traditional programming : humans write the RULES, the program applies them to data.
Machine learning        : we give DATA + ANSWERS (Training sheet), the model learns the rules.

Input  : input_data.xlsx  -> sheets 'Training' (with Approved) and 'Input' (new applicants)
Output : written back to the same file, sheet 'Output'
"""
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

FILE = "input_data.xlsx"
FEATURES = ["MonthlyIncome", "CreditScore", "ExistingEMI", "LoanAmount"]


# ---------- 1. TRADITIONAL PROGRAMMING (hand-written rules) ----------
def traditional_decision(row):
    if row["CreditScore"] < 700:
        return "No"
    if row["ExistingEMI"] > 0.40 * row["MonthlyIncome"]:
        return "No"
    if row["LoanAmount"] > 20 * row["MonthlyIncome"]:
        return "No"
    return "Yes"


# ---------- 2. MACHINE LEARNING (rules learned from data) ----------
def train_model(train_df):
    X, y = train_df[FEATURES], train_df["Approved"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=1)
    model = DecisionTreeClassifier(max_depth=4, random_state=1).fit(X_tr, y_tr)
    acc = accuracy_score(y_te, model.predict(X_te))
    model.fit(X, y)  # refit on all data
    return model, acc


def main():
    train_df = pd.read_excel(FILE, sheet_name="Training")
    input_df = pd.read_excel(FILE, sheet_name="Input")

    out = input_df.copy()
    out["Traditional_Decision"] = input_df.apply(traditional_decision, axis=1)

    model, acc = train_model(train_df)
    out["ML_Decision"] = model.predict(input_df[FEATURES])
    out["Same_Result"] = (out["Traditional_Decision"] == out["ML_Decision"]).map({True: "Yes", False: "No"})

    # write back to the SAME Excel file (other sheets stay untouched)
    with pd.ExcelWriter(FILE, engine="openpyxl", mode="a", if_sheet_exists="replace") as w:
        out.to_excel(w, sheet_name="Output", index=False)

    print(f"ML test accuracy on held-out training data: {acc:.2%}")
    print(f"Traditional vs ML agree on {(out['Same_Result'] == 'Yes').sum()}/{len(out)} applicants")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
