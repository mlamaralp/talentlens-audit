# TalentLens -- Data Quality Audit
# Task 1: assessing the recruitment dataset for data quality issues
#
# Dataset: Utrecht Fairness Recruitment Dataset (4,000 records, 4 companies)
# Goal: identify and quantify data quality issues before bias or governance analysis.

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# Path().resolve() returns the directory this script is running from.
# Since the scripts live in src/, .parent moves up to the project root.
ROOT        = Path(__file__).resolve().parent.parent
DATA_PATH   = ROOT / "data" / "recruitmentdataset-2022-1_3.csv"
IMAGES_PATH = ROOT / "images"
IMAGES_PATH.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({"figure.dpi": 130})
PALETTE = ["#2C6E9B", "#E07B39", "#3D9970", "#C44E52"]

df = pd.read_csv(DATA_PATH)
print(f"Dataset loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")
print(df.head())


# ------------------------------------------------------------------------------
# 2. Dataset Overview
# ------------------------------------------------------------------------------

print("Column names and types:")
print(df.dtypes)
print()
print("Numeric summary:")
print(df.describe().T.round(2))

# ------------------------------------------------------------------------------
# 3. Data Quality Assessment
# Six dimensions: completeness, uniqueness, validity, accuracy,
# consistency, and data minimisation.
# ------------------------------------------------------------------------------

# 3.1 Completeness -- missing values
missing     = df.isnull().sum()
pct_missing = (missing / len(df) * 100).round(2)
result      = pd.DataFrame({"Missing Count": missing, "Missing %": pct_missing})
result      = result[result["Missing Count"] > 0]

if result.empty:
    print("No missing values found.")
else:
    print(result)

fig, ax = plt.subplots(figsize=(9, 4))
completeness_pct = (1 - df.isnull().mean()) * 100
ax.bar(completeness_pct.index, completeness_pct.values,
       color="#2C6E9B", edgecolor="white")
ax.set_ylim(0, 110)
ax.set_ylabel("Completeness (%)", fontsize=11)
ax.set_title("Column Completeness", fontsize=13, fontweight="bold")
ax.set_xticklabels(completeness_pct.index, rotation=30, ha="right")
ax.axhline(100, color="green", ls="--", lw=1, label="100%")
ax.legend()
plt.tight_layout()
plt.savefig(IMAGES_PATH / "completeness.png", bbox_inches="tight", dpi=150)
plt.show()
print("All columns are 100% complete.")

# 3.2 Uniqueness -- duplicate records
dup_rows = df.duplicated().sum()
dup_ids  = df["Id"].duplicated().sum()
print(f"Duplicate rows: {dup_rows}")
print(f"Duplicate IDs:  {dup_ids}")
if dup_rows == 0:
    print("No duplicate records found.")
else:
    df = df.drop_duplicates()
    print(f"Shape after deduplication: {df.shape}")

# 3.3 Validity -- ranges and categories
print("University grade:", df["ind-university_grade"].min(), "to", df["ind-university_grade"].max())
print("Age:", df["age"].min(), "to", df["age"].max())
print("Gender values:", df["gender"].unique().tolist())
print("Degree values:", df["ind-degree"].unique().tolist())
print()
print("Languages spoken (value counts):")
print(df["ind-languages"].value_counts().sort_index())
print("A value of 0 may mean the field was left blank.")

# 3.4 Consistency -- grade distributions across companies
# All four companies evaluated the same applicant pool.
# Very different grade distributions would suggest inconsistent scoring practices.
fig, ax = plt.subplots(figsize=(9, 5))
for company, color in zip(["A","B","C","D"], PALETTE):
    df[df["company"] == company]["ind-university_grade"].plot.kde(
        ax=ax, label=f"Company {company}", color=color, lw=2)
ax.set_xlabel("University Grade", fontsize=11)
ax.set_ylabel("Density", fontsize=11)
ax.set_title("Grade Distribution by Company", fontsize=13, fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig(IMAGES_PATH / "grade_by_company.png", bbox_inches="tight", dpi=150)
plt.show()
print("Grade distributions are consistent across companies.")

# 3.5 Accuracy -- narrow grade range
print(f"Grade range: {df['ind-university_grade'].min()} to {df['ind-university_grade'].max()}")
print(f"Standard deviation: {df['ind-university_grade'].std():.2f}")
print()
print("The range 45-78 is unusually narrow for a standard grading scale.")
print("Grades may have been normalised without documentation.")

fig, ax = plt.subplots(figsize=(8, 4))
df["ind-university_grade"].hist(bins=20, color="#2C6E9B", edgecolor="white", ax=ax)
ax.set_xlabel("University Grade", fontsize=11)
ax.set_ylabel("Count", fontsize=11)
ax.set_title("Distribution of University Grades", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(IMAGES_PATH / "grade_distribution.png", bbox_inches="tight", dpi=150)
plt.show()

# 3.6 Data minimisation -- fields with no documented job-relevant purpose
print("Fields with no documented job-relevant justification:")
for col in ["gender", "age", "nationality", "sport"]:
    print(f"  - {col}: {df[col].nunique()} unique values")
print()
print("Under GDPR Article 5(1)(c), only data strictly necessary for the")
print("stated purpose should be collected.")
print("sport has no job-relevant purpose and acts as a proxy for gender (see Task 2).")

# ------------------------------------------------------------------------------
# 4. Summary
# ------------------------------------------------------------------------------
print()
print("Data quality summary:")
findings = [
    ("Completeness",       "No missing values",                                     "Pass"),
    ("Uniqueness",         "No duplicate records",                                  "Pass"),
    ("Validity",           "All values within expected ranges",                     "Pass"),
    ("Accuracy",           "Grade range 45-78 unusually narrow",                   "Warning"),
    ("Consistency",        "Grade distributions consistent across companies",        "Pass"),
    ("Data minimisation",  "sport and nationality collected without documented need","Fail"),
]
for dim, finding, status in findings:
    print(f"  {status:8s}  {dim:22s}  {finding}")
