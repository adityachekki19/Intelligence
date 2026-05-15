"""
Attendance Analytics System
Uses only:
- numpy
- pandas
- python standard library

Run:
python app.py
"""

import os
import glob
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


# ──────────────────────────────────────────────
# 1. SAMPLE DATA GENERATOR
# ──────────────────────────────────────────────

def generate_sample_data(output_dir="attendance_data"):

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/daily", exist_ok=True)

    random.seed(42)
    np.random.seed(42)

    students = pd.DataFrame({
        "Student_ID": [f"S{i:03d}" for i in range(1, 21)],

        "Name": [
            "Ravi", "Sneha", "Rahul", "Priya", "Amit",
            "Kavita", "Suresh", "Meena", "Arjun", "Divya",
            "Nikhil", "Pooja", "Sanjay", "Aarti", "Vikram",
            "Nisha", "Rohan", "Sunita", "Karthik", "Lata"
        ],

        "College": [
            "ABC College", "XYZ College", "ABC College",
            "LMN College", "XYZ College",

            "LMN College", "ABC College", "XYZ College",
            "LMN College", "ABC College",

            "XYZ College", "ABC College", "LMN College",
            "XYZ College", "ABC College",

            "LMN College", "XYZ College", "ABC College",
            "LMN College", "XYZ College"
        ],

        "Department": [
            "CSE", "ECE", "ME", "CSE", "ECE",
            "ME", "CSE", "ECE", "ME", "CSE",
            "ECE", "ME", "CSE", "ECE", "ME",
            "CSE", "ECE", "ME", "CSE", "ECE"
        ],

        "Year": np.random.choice([1, 2, 3, 4], size=20)
    })

    students.to_csv(f"{output_dir}/students.csv", index=False)

    subjects = ["Machine Learning", "Deep Learning", "Python"]

    start_date = datetime(2025, 2, 1)

    base_prob = np.random.uniform(0.55, 0.98, size=len(students))

    for day_offset in range(10):

        date = start_date + timedelta(days=day_offset)

        subject = subjects[day_offset % len(subjects)]

        if subject == "Deep Learning" and day_offset % 4 == 1:
            probs = np.clip(base_prob * 0.5, 0.1, 0.6)
        else:
            probs = base_prob

        statuses = [
            "Present" if np.random.rand() < p else "Absent"
            for p in probs
        ]

        day_df = pd.DataFrame({
            "Date": date.strftime("%Y-%m-%d"),
            "Student_ID": students["Student_ID"],
            "Subject": subject,
            "Status": statuses
        })

        day_df.to_csv(
            f"{output_dir}/daily/attendance_day{day_offset+1}.csv",
            index=False
        )

    print("\nSample CSV files generated successfully.\n")


# ──────────────────────────────────────────────
# 2. LOAD AND MERGE DATA
# ──────────────────────────────────────────────

def load_and_merge(data_dir):

    students_path = f"{data_dir}/students.csv"

    if not os.path.exists(students_path):
        raise FileNotFoundError("students.csv not found")

    students = pd.read_csv(students_path)

    csv_files = sorted(
        glob.glob(f"{data_dir}/daily/*.csv")
    )

    if not csv_files:
        raise FileNotFoundError("No CSV attendance files found")

    df_list = []

    for file in csv_files:

        try:
            data = pd.read_csv(file)
            df_list.append(data)

        except Exception as e:
            print(f"Error reading {file}: {e}")

    attendance = pd.concat(df_list, ignore_index=True)

    attendance["Present"] = attendance["Status"].apply(
        lambda x: 1 if str(x).strip().lower() == "present" else 0
    )

    df = attendance.merge(
        students,
        on="Student_ID",
        how="left"
    )

    df["Date"] = pd.to_datetime(df["Date"])

    return df


# ──────────────────────────────────────────────
# 3. ANALYTICS FUNCTIONS
# ──────────────────────────────────────────────

def student_attendance(df):

    grp = df.groupby(["Student_ID", "Name"])

    total = grp["Present"].count().rename("Total_Classes")

    present = grp["Present"].sum().rename("Present_Count")

    report = pd.concat([total, present], axis=1)

    report["Absent_Count"] = (
        report["Total_Classes"] - report["Present_Count"]
    )

    report["Attendance_Pct"] = (
        report["Present_Count"] /
        report["Total_Classes"] * 100
    ).round(2)

    return report.reset_index().sort_values("Attendance_Pct")


def low_attendance_students(df, threshold=75):

    report = student_attendance(df)

    return report[
        report["Attendance_Pct"] < threshold
    ]


def department_attendance(df):

    report = (
        df.groupby("Department")["Present"]
        .mean()
        .reset_index()
    )

    report["Attendance_Pct"] = (
        report["Present"] * 100
    ).round(2)

    return report.sort_values(
        "Attendance_Pct",
        ascending=False
    )


def subject_attendance(df):

    report = (
        df.groupby("Subject")["Present"]
        .mean()
        .reset_index()
    )

    report["Attendance_Pct"] = (
        report["Present"] * 100
    ).round(2)

    return report.sort_values(
        "Attendance_Pct",
        ascending=False
    )


def daily_trend(df):

    report = (
        df.groupby("Date")["Present"]
        .mean()
        .reset_index()
    )

    report["Attendance_Pct"] = (
        report["Present"] * 100
    ).round(2)

    return report


# ──────────────────────────────────────────────
# 4. REPORT EXPORT
# ──────────────────────────────────────────────

def export_reports(df):

    os.makedirs("reports", exist_ok=True)

    student_attendance(df).to_csv(
        "reports/student_report.csv",
        index=False
    )

    department_attendance(df).to_csv(
        "reports/department_report.csv",
        index=False
    )

    subject_attendance(df).to_csv(
        "reports/subject_report.csv",
        index=False
    )

    daily_trend(df).to_csv(
        "reports/daily_trend.csv",
        index=False
    )

    print("\nReports exported successfully.\n")


# ──────────────────────────────────────────────
# 5. DISPLAY FUNCTIONS
# ──────────────────────────────────────────────

def show_overview(df):

    print("\n========== OVERVIEW ==========\n")

    print(f"Total Students      : {df['Student_ID'].nunique()}")

    print(f"Total Records       : {len(df)}")

    print(f"Average Attendance  : {round(df['Present'].mean()*100, 2)}%")


def show_student_report(df):

    print("\n========== STUDENT REPORT ==========\n")

    print(student_attendance(df).to_string(index=False))


def show_low_attendance(df):

    print("\n========== LOW ATTENDANCE ==========\n")

    low = low_attendance_students(df)

    if low.empty:
        print("No low attendance students")
    else:
        print(low.to_string(index=False))


def show_department_report(df):

    print("\n========== DEPARTMENT REPORT ==========\n")

    print(department_attendance(df).to_string(index=False))


def show_subject_report(df):

    print("\n========== SUBJECT REPORT ==========\n")

    print(subject_attendance(df).to_string(index=False))


def show_daily_trend(df):

    print("\n========== DAILY TREND ==========\n")

    print(daily_trend(df).to_string(index=False))


# ──────────────────────────────────────────────
# 6. MAIN MENU
# ──────────────────────────────────────────────

def menu(df):

    while True:

        print("""
========================================
ATTENDANCE ANALYTICS SYSTEM
========================================

1. Overview
2. Student Attendance Report
3. Low Attendance Students
4. Department Analytics
5. Subject Analytics
6. Daily Attendance Trend
7. Export Reports
0. Exit
""")

        choice = input("Enter option: ").strip()

        if choice == "1":
            show_overview(df)

        elif choice == "2":
            show_student_report(df)

        elif choice == "3":
            show_low_attendance(df)

        elif choice == "4":
            show_department_report(df)

        elif choice == "5":
            show_subject_report(df)

        elif choice == "6":
            show_daily_trend(df)

        elif choice == "7":
            export_reports(df)

        elif choice == "0":
            print("\nExiting...\n")
            break

        else:
            print("\nInvalid option\n")

        input("\nPress Enter to continue...")


# ──────────────────────────────────────────────
# 7. MAIN FUNCTION
# ──────────────────────────────────────────────

def main():

    print("\nATTENDANCE ANALYTICS SYSTEM\n")

    data_dir = "attendance_data"

    if not os.path.exists(f"{data_dir}/students.csv"):

        print("Generating sample data...")

        generate_sample_data(data_dir)

    print("Loading data...\n")

    df = load_and_merge(data_dir)

    print(f"Loaded {len(df)} attendance records.\n")

    menu(df)


# ──────────────────────────────────────────────
# 8. ENTRY POINT
# ──────────────────────────────────────────────

if __name__ == "__main__":
    main()
