"""
Attendance Analytics System
Uses only: numpy, pandas, python stdlib
Run: python app.py
"""

import os
import glob
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random


# ──────────────────────────────────────────────
# 1.  SAMPLE DATA GENERATOR
# ──────────────────────────────────────────────

def generate_sample_data(output_dir: str = "attendance_data") -> str:
    """
    Creates synthetic student master CSV + daily attendance XLSX files
    so the pipeline works out-of-the-box without any real uploads.
    Returns the path to the data directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "daily"), exist_ok=True)

    random.seed(42)
    np.random.seed(42)

    # --- Student master ---
    students = pd.DataFrame({
        "Student_ID": [f"S{i:03d}" for i in range(1, 21)],
        "Name": [
            "Ravi",    "Sneha",  "Rahul",  "Priya",   "Amit",
            "Kavita",  "Suresh", "Meena",  "Arjun",   "Divya",
            "Nikhil",  "Pooja",  "Sanjay", "Aarti",   "Vikram",
            "Nisha",   "Rohan",  "Sunita", "Karthik",  "Lata",
        ],
        "College": [
            "ABC College", "XYZ College", "ABC College", "LMN College", "XYZ College",
            "LMN College", "ABC College", "XYZ College", "LMN College", "ABC College",
            "XYZ College", "ABC College", "LMN College", "XYZ College", "ABC College",
            "LMN College", "XYZ College", "ABC College", "LMN College", "XYZ College",
        ],
        "Department": [
            "CSE", "ECE", "ME",  "CSE", "ECE",
            "ME",  "CSE", "ECE", "ME",  "CSE",
            "ECE", "ME",  "CSE", "ECE", "ME",
            "CSE", "ECE", "ME",  "CSE", "ECE",
        ],
        "Year": np.random.choice([1, 2, 3, 4], size=20),
    })
    students_path = os.path.join(output_dir, "students.csv")
    students.to_csv(students_path, index=False)
    print(f"  [+] Generated {students_path}")

    # --- Daily attendance (10 days, 3 subjects) ---
    subjects = ["Machine Learning", "Deep Learning", "Python"]
    start_date = datetime(2025, 2, 1)

    # Assign each student a base attendance probability
    base_prob = np.random.uniform(0.55, 0.98, size=len(students))

    for day_offset in range(10):
        date = start_date + timedelta(days=day_offset)
        date_str = date.strftime("%Y-%m-%d")
        subject = subjects[day_offset % len(subjects)]

        # Deep Learning occasionally has very low attendance
        if subject == "Deep Learning" and day_offset % 4 == 1:
            probs = np.clip(base_prob * 0.5, 0.1, 0.6)
        else:
            probs = base_prob

        statuses = [
            "Present" if np.random.rand() < p else "Absent"
            for p in probs
        ]

        day_df = pd.DataFrame({
            "Date":       date_str,
            "Student_ID": students["Student_ID"].values,
            "Subject":    subject,
            "Status":     statuses,
        })

        file_path = os.path.join(
            output_dir, "daily", f"attendance_day{day_offset+1}.xlsx"
        )
        day_df.to_excel(file_path, index=False)

    print(f"  [+] Generated 10 daily attendance files in {output_dir}/daily/")
    return output_dir


# ──────────────────────────────────────────────
# 2.  DATA LOADING & INTEGRATION
# ──────────────────────────────────────────────

def load_and_merge(data_dir: str) -> pd.DataFrame:
    """Load all attendance xlsx files, concatenate, merge with students."""

    # Load student master
    students_path = os.path.join(data_dir, "students.csv")
    if not os.path.exists(students_path):
        raise FileNotFoundError(f"students.csv not found in {data_dir}")
    students = pd.read_csv(students_path)

    # Load all daily attendance files
    pattern = os.path.join(data_dir, "daily", "*.xlsx")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No attendance xlsx files found at {pattern}")

    df_list = []
    for f in sorted(files):
        data = pd.read_excel(f)
        df_list.append(data)
    attendance = pd.concat(df_list, ignore_index=True)

    # Numeric flag
    attendance["Present"] = attendance["Status"].apply(
        lambda x: 1 if str(x).strip().lower() == "present" else 0
    )

    # Merge
    df = attendance.merge(students, on="Student_ID", how="left")
    df["Date"] = pd.to_datetime(df["Date"])
    return df


# ──────────────────────────────────────────────
# 3.  ANALYTICS FUNCTIONS
# ──────────────────────────────────────────────

def student_attendance(df: pd.DataFrame) -> pd.DataFrame:
    """Student-wise attendance report."""
    grp = df.groupby(["Student_ID", "Name"])
    total   = grp["Present"].count().rename("Total_Classes")
    present = grp["Present"].sum().rename("Present_Count")
    report  = pd.concat([total, present], axis=1)
    report["Absent_Count"]   = report["Total_Classes"] - report["Present_Count"]
    report["Attendance_Pct"] = (report["Present_Count"] / report["Total_Classes"] * 100).round(2)
    return report.reset_index().sort_values("Attendance_Pct")


def low_attendance_students(df: pd.DataFrame, threshold: float = 75.0) -> pd.DataFrame:
    """Students below attendance threshold."""
    report = student_attendance(df)
    return report[report["Attendance_Pct"] < threshold]


def department_attendance(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("Department")["Present"]
          .agg(["mean", "count"])
          .rename(columns={"mean": "Avg_Attendance_Pct", "count": "Total_Records"})
          .assign(Avg_Attendance_Pct=lambda x: (x["Avg_Attendance_Pct"] * 100).round(2))
          .reset_index()
          .sort_values("Avg_Attendance_Pct", ascending=False)
    )


def college_attendance(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("College")["Present"]
          .agg(["mean", "count"])
          .rename(columns={"mean": "Avg_Attendance_Pct", "count": "Total_Records"})
          .assign(Avg_Attendance_Pct=lambda x: (x["Avg_Attendance_Pct"] * 100).round(2))
          .reset_index()
          .sort_values("Avg_Attendance_Pct", ascending=False)
    )


def subject_attendance(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("Subject")["Present"]
          .agg(["mean", "count"])
          .rename(columns={"mean": "Avg_Attendance_Pct", "count": "Total_Records"})
          .assign(Avg_Attendance_Pct=lambda x: (x["Avg_Attendance_Pct"] * 100).round(2))
          .reset_index()
          .sort_values("Avg_Attendance_Pct", ascending=False)
    )


def daily_trend(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("Date")["Present"]
          .mean()
          .reset_index()
          .rename(columns={"Present": "Avg_Attendance_Pct"})
          .assign(Avg_Attendance_Pct=lambda x: (x["Avg_Attendance_Pct"] * 100).round(2))
          .sort_values("Date")
    )


def missing_class_detection(df: pd.DataFrame, threshold: float = 60.0) -> pd.DataFrame:
    """Classes (Date × Subject) where attendance dropped below threshold."""
    class_att = (
        df.groupby(["Date", "Subject"])["Present"]
          .mean()
          .reset_index()
          .rename(columns={"Present": "Attendance_Pct"})
          .assign(Attendance_Pct=lambda x: (x["Attendance_Pct"] * 100).round(2))
    )
    return class_att[class_att["Attendance_Pct"] < threshold].sort_values("Attendance_Pct")


# ──────────────────────────────────────────────
# 4.  ASCII CHART HELPERS
# ──────────────────────────────────────────────

def _bar(value: float, max_val: float = 100.0, width: int = 30) -> str:
    filled = int(round(value / max_val * width))
    return "█" * filled + "░" * (width - filled)


def print_bar_chart(df: pd.DataFrame, label_col: str, value_col: str, title: str):
    print(f"\n{'═'*60}")
    print(f"  {title}")
    print(f"{'═'*60}")
    max_val = df[value_col].max()
    for _, row in df.iterrows():
        bar  = _bar(row[value_col], max_val)
        lbl  = str(row[label_col])[:18].ljust(18)
        print(f"  {lbl} {bar} {row[value_col]:.1f}%")


def print_line_chart(series: pd.Series, title: str, height: int = 8):
    """Minimal ASCII line chart."""
    values = series.values.astype(float)
    lo, hi = values.min(), values.max()
    rng = hi - lo if hi != lo else 1

    print(f"\n{'═'*60}")
    print(f"  {title}")
    print(f"{'═'*60}")

    for row_i in range(height, -1, -1):
        threshold = lo + (row_i / height) * rng
        line = f"  {threshold:5.1f}% | "
        for v in values:
            if v >= threshold:
                line += "▓ "
            else:
                line += "  "
        print(line)

    print("         " + "──" * len(values))
    indices = [str(i + 1) for i in range(len(values))]
    print("         " + " ".join(f"{x:<2}" for x in indices))
    print("         (day index)")


# ──────────────────────────────────────────────
# 5.  INSIGHTS GENERATOR
# ──────────────────────────────────────────────

def generate_insights(df: pd.DataFrame, threshold: float = 75.0) -> list[str]:
    insights = []

    overall = df["Present"].mean() * 100
    insights.append(f"Overall average attendance: {overall:.1f}%")

    low = low_attendance_students(df, threshold)
    insights.append(
        f"{len(low)} student(s) have attendance below {threshold}%: "
        + ", ".join(low["Name"].tolist())
    )

    dept = department_attendance(df)
    best_dept  = dept.iloc[0]
    worst_dept = dept.iloc[-1]
    diff = best_dept["Avg_Attendance_Pct"] - worst_dept["Avg_Attendance_Pct"]
    insights.append(
        f"{worst_dept['Department']} dept attendance is {diff:.1f}% lower "
        f"than {best_dept['Department']}"
    )

    subj = subject_attendance(df)
    worst_subj = subj.iloc[-1]
    insights.append(
        f"'{worst_subj['Subject']}' has the lowest attendance at {worst_subj['Avg_Attendance_Pct']:.1f}%"
    )

    col = college_attendance(df)
    worst_col = col.iloc[-1]
    insights.append(
        f"'{worst_col['College']}' has the highest absentee rate "
        f"(attendance: {worst_col['Avg_Attendance_Pct']:.1f}%)"
    )

    missing = missing_class_detection(df, threshold=60.0)
    if not missing.empty:
        row = missing.iloc[0]
        insights.append(
            f"Possible missing/skipped class: {row['Subject']} on "
            f"{row['Date'].strftime('%Y-%m-%d')} had only {row['Attendance_Pct']:.1f}% attendance"
        )

    return insights


# ──────────────────────────────────────────────
# 6.  REPORT EXPORT
# ──────────────────────────────────────────────

def export_reports(df: pd.DataFrame, out_dir: str = "reports"):
    os.makedirs(out_dir, exist_ok=True)

    student_attendance(df).to_csv(
        os.path.join(out_dir, "student_report.csv"), index=False
    )
    department_attendance(df).to_csv(
        os.path.join(out_dir, "department_report.csv"), index=False
    )
    college_attendance(df).to_csv(
        os.path.join(out_dir, "college_report.csv"), index=False
    )
    subject_attendance(df).to_csv(
        os.path.join(out_dir, "subject_report.csv"), index=False
    )
    daily_trend(df).to_csv(
        os.path.join(out_dir, "daily_trend.csv"), index=False
    )
    missing_class_detection(df).to_csv(
        os.path.join(out_dir, "missing_classes.csv"), index=False
    )
    print(f"\n  [+] CSV reports saved to '{out_dir}/' directory")


# ──────────────────────────────────────────────
# 7.  INTERACTIVE MENU
# ──────────────────────────────────────────────

def print_table(df: pd.DataFrame, max_rows: int = 20):
    print(df.head(max_rows).to_string(index=False))


def interactive_menu(df: pd.DataFrame):
    menu = """
╔══════════════════════════════════════╗
║   ATTENDANCE ANALYTICS SYSTEM        ║
╚══════════════════════════════════════╝
  1. Overview (totals & averages)
  2. Student attendance report
  3. Low attendance students
  4. Department-wise analytics
  5. College-wise analytics
  6. Subject-wise analytics
  7. Daily attendance trend (chart)
  8. Missing class detection
  9. Generate insights
 10. Export all CSV reports
  0. Exit
"""
    while True:
        print(menu)
        choice = input("  Enter option: ").strip()

        if choice == "0":
            print("\n  Goodbye!\n")
            break

        elif choice == "1":
            total_students = df["Student_ID"].nunique()
            total_classes  = df.groupby(["Date", "Subject"]).ngroups
            avg_att        = df["Present"].mean() * 100
            low_count      = len(low_attendance_students(df))
            print(f"\n  Total Students       : {total_students}")
            print(f"  Total Class Sessions : {total_classes}")
            print(f"  Overall Avg Att.     : {avg_att:.1f}%")
            print(f"  Students Below 75%   : {low_count}")

        elif choice == "2":
            rpt = student_attendance(df)
            print("\n")
            print_table(rpt)

        elif choice == "3":
            t = input("  Threshold % (default 75): ").strip()
            threshold = float(t) if t else 75.0
            low = low_attendance_students(df, threshold)
            if low.empty:
                print(f"\n  No students below {threshold}%")
            else:
                print(f"\n  Students below {threshold}%:")
                print_table(low)

        elif choice == "4":
            dept = department_attendance(df)
            print_table(dept)
            print_bar_chart(dept, "Department", "Avg_Attendance_Pct",
                            "Department-wise Attendance")

        elif choice == "5":
            col = college_attendance(df)
            print_table(col)
            print_bar_chart(col, "College", "Avg_Attendance_Pct",
                            "College-wise Attendance")

        elif choice == "6":
            subj = subject_attendance(df)
            print_table(subj)
            print_bar_chart(subj, "Subject", "Avg_Attendance_Pct",
                            "Subject-wise Attendance")

        elif choice == "7":
            trend = daily_trend(df)
            print_table(trend)
            print_line_chart(trend["Avg_Attendance_Pct"],
                             "Daily Attendance Trend")

        elif choice == "8":
            t = input("  Alert threshold % (default 60): ").strip()
            threshold = float(t) if t else 60.0
            missing = missing_class_detection(df, threshold)
            if missing.empty:
                print(f"\n  No classes below {threshold}%")
            else:
                print(f"\n  Classes with attendance < {threshold}%:")
                print_table(missing)

        elif choice == "9":
            insights = generate_insights(df)
            print("\n  ── KEY INSIGHTS ──────────────────────────")
            for i, insight in enumerate(insights, 1):
                print(f"  {i}. {insight}")

        elif choice == "10":
            export_reports(df)

        else:
            print("  Invalid option, try again.")

        input("\n  Press Enter to continue...")


# ──────────────────────────────────────────────
# 8.  ENTRY POINT
# ──────────────────────────────────────────────

def main():
    print("\n" + "="*60)
    print("  ATTENDANCE ANALYTICS SYSTEM")
    print("="*60)

    data_dir = "attendance_data"

    # Generate sample data if none exists
    if not os.path.exists(os.path.join(data_dir, "students.csv")):
        print("\n  No data found. Generating sample data...\n")
        generate_sample_data(data_dir)

    print("\n  Loading and merging data...")
    df = load_and_merge(data_dir)
    print(f"  Loaded {len(df)} attendance records for "
          f"{df['Student_ID'].nunique()} students.\n")

    interactive_menu(df)


if __name__ == "__main__":
    main()
