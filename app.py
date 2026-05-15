import streamlit as st
import pandas as pd
import numpy as np
import os
import glob
import random
from datetime import datetime, timedelta

st.set_page_config(
    page_title="Attendance Analytics System",
    layout="wide"
)

# ──────────────────────────────────────────────
# SAMPLE DATA GENERATOR
# ──────────────────────────────────────────────

@st.cache_data
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

    return output_dir


# ──────────────────────────────────────────────
# LOAD DATA
# ──────────────────────────────────────────────

@st.cache_data
def load_and_merge(data_dir):

    students_path = f"{data_dir}/students.csv"

    if not os.path.exists(students_path):
        return pd.DataFrame()

    students = pd.read_csv(students_path)

    csv_files = sorted(
        glob.glob(f"{data_dir}/daily/*.csv")
    )

    if len(csv_files) == 0:
        return pd.DataFrame()

    df_list = []

    for file in csv_files:

        try:
            data = pd.read_csv(file)
            df_list.append(data)

        except Exception as e:
            st.error(f"Error reading {file}: {e}")

    if len(df_list) == 0:
        return pd.DataFrame()

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
# ANALYTICS FUNCTIONS
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
# MAIN APP
# ──────────────────────────────────────────────

st.title("📊 Attendance Analytics System")

data_dir = "attendance_data"

# Generate sample data if missing
if not os.path.exists(data_dir):
    generate_sample_data(data_dir)

if not os.path.exists(f"{data_dir}/students.csv"):
    generate_sample_data(data_dir)

if not os.path.exists(f"{data_dir}/daily"):
    generate_sample_data(data_dir)

csv_files = glob.glob(f"{data_dir}/daily/*.csv")

if len(csv_files) == 0:
    generate_sample_data(data_dir)

# Load data
df = load_and_merge(data_dir)

if df.empty:
    st.error("No data found")
    st.stop()

# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────

st.sidebar.title("Menu")

option = st.sidebar.selectbox(
    "Select Analytics",
    [
        "Overview",
        "Student Attendance",
        "Low Attendance",
        "Department Analytics",
        "Subject Analytics",
        "Daily Trend"
    ]
)

# ──────────────────────────────────────────────
# OVERVIEW
# ──────────────────────────────────────────────

if option == "Overview":

    st.header("Overview")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Total Students",
        df["Student_ID"].nunique()
    )

    c2.metric(
        "Total Records",
        len(df)
    )

    c3.metric(
        "Average Attendance %",
        round(df["Present"].mean() * 100, 2)
    )

# ──────────────────────────────────────────────
# STUDENT ATTENDANCE
# ──────────────────────────────────────────────

elif option == "Student Attendance":

    st.header("Student Attendance Report")

    report = student_attendance(df)

    st.dataframe(report, use_container_width=True)

# ──────────────────────────────────────────────
# LOW ATTENDANCE
# ──────────────────────────────────────────────

elif option == "Low Attendance":

    st.header("Low Attendance Students")

    threshold = st.slider(
        "Attendance Threshold %",
        0,
        100,
        75
    )

    low = low_attendance_students(df, threshold)

    st.dataframe(low, use_container_width=True)

# ──────────────────────────────────────────────
# DEPARTMENT ANALYTICS
# ──────────────────────────────────────────────

elif option == "Department Analytics":

    st.header("Department Attendance")

    dept = department_attendance(df)

    st.dataframe(dept, use_container_width=True)

    st.bar_chart(
        dept.set_index("Department")["Attendance_Pct"]
    )

# ──────────────────────────────────────────────
# SUBJECT ANALYTICS
# ──────────────────────────────────────────────

elif option == "Subject Analytics":

    st.header("Subject Attendance")

    subject = subject_attendance(df)

    st.dataframe(subject, use_container_width=True)

    st.bar_chart(
        subject.set_index("Subject")["Attendance_Pct"]
    )

# ──────────────────────────────────────────────
# DAILY TREND
# ──────────────────────────────────────────────

elif option == "Daily Trend":

    st.header("Daily Attendance Trend")

    trend = daily_trend(df)

    st.dataframe(trend, use_container_width=True)

    st.line_chart(
        trend.set_index("Date")["Attendance_Pct"]
    )
