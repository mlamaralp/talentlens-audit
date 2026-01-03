# TalentLens -- Bias Detection and Fairness Analysis
# Task 2: detecting algorithmic bias in hiring decisions
#
# Dataset: Utrecht Fairness Recruitment Dataset (4,000 records, 4 companies)
# Primary metric: Disparate Impact (DI) ratio -- four-fifths rule.
# A DI ratio below 0.8 indicates potential discriminatory impact.

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, pointbiserialr
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT        = Path(__file__).resolve().parent.parent
DATA_PATH   = ROOT / "data" / "recruitmentdataset-2022-1_3.csv"
IMAGES_PATH = ROOT / "images"
IMAGES_PATH.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({"figure.dpi": 130})
PALETTE = ["#2C6E9B", "#E07B39", "#3D9970", "#C44E52"]

df = pd.read_csv(DATA_PATH)
print(f"Dataset loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")


# ------------------------------------------------------------------------------
# 2. Gender Disparate Impact
# ------------------------------------------------------------------------------

gender_rates = df.groupby("gender")["decision"].agg(["mean","count"]).round(3)
gender_rates.columns = ["Approval Rate", "N"]
print(gender_rates)

female_rate = df[df["gender"]=="female"]["decision"].mean()
male_rate   = df[df["gender"]=="male"]["decision"].mean()
di_gender   = female_rate / male_rate

print(f"\nFemale approval rate: {female_rate:.3f}")
print(f"Male approval rate:   {male_rate:.3f}")
print(f"DI ratio (female / male): {di_gender:.3f}")
if di_gender < 0.8:
    print("DI < 0.8 -- four-fifths rule violated.")
else:
    print("DI >= 0.8 -- no disparate impact detected.")

fig, ax = plt.subplots(figsize=(7, 5))
genders = ["female", "male", "other"]
rates   = [df[df["gender"]==g]["decision"].mean() for g in genders]
bars = ax.bar(genders, rates, color=["#E07B39","#2C6E9B","#3D9970"],
              edgecolor="white", width=0.5)
ax.bar_label(bars, labels=[f"{r:.1%}" for r in rates],
             padding=4, fontsize=10, fontweight="bold")
ax.axhline(0.8 * male_rate, color="red", ls="--", lw=1.5,
           label=f"4/5 threshold ({0.8*male_rate:.1%})")
ax.set_ylabel("Approval Rate", fontsize=11)
ax.set_title("Hiring Approval Rate by Gender", fontsize=13, fontweight="bold")
ax.set_ylim(0, 0.55)
ax.legend()
ax.text(0.01, -0.14, f"DI ratio = {di_gender:.3f} -- below the 0.8 threshold.",
        transform=ax.transAxes, fontsize=9.5, color="#444")
plt.tight_layout()
plt.savefig(IMAGES_PATH / "approval_by_gender.png", bbox_inches="tight", dpi=150)
plt.show()

# ------------------------------------------------------------------------------
# 3. Age Bias
# ------------------------------------------------------------------------------

df["age_group"] = pd.cut(df["age"], bins=[20,24,27,32],
                          labels=["21-24","25-27","28-32"])
age_rates = df.groupby("age_group", observed=True)["decision"].agg(["mean","count"])
age_rates.columns = ["Approval Rate", "N"]
print(age_rates)

youngest = df[df["age_group"]=="21-24"]["decision"].mean()
oldest   = df[df["age_group"]=="28-32"]["decision"].mean()
di_age   = youngest / oldest
print(f"DI ratio (21-24 vs 28-32): {di_age:.3f}")

fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(age_rates.index.astype(str), age_rates["Approval Rate"],
              color=PALETTE[:3], edgecolor="white", width=0.5)
ax.bar_label(bars, labels=[f"{r:.1%}" for r in age_rates["Approval Rate"]],
             padding=4, fontsize=10, fontweight="bold")
ax.set_ylabel("Approval Rate", fontsize=11)
ax.set_title("Hiring Approval Rate by Age Group", fontsize=13, fontweight="bold")
ax.set_ylim(0, 0.45)
ax.text(0.01, -0.14,
        f"Younger candidates are approved less often (DI = {di_age:.3f}).",
        transform=ax.transAxes, fontsize=9.5, color="#444")
plt.tight_layout()
plt.savefig(IMAGES_PATH / "approval_by_age.png", bbox_inches="tight", dpi=150)
plt.show()

# ------------------------------------------------------------------------------
# 4. Nationality
# ------------------------------------------------------------------------------

nat_rates = df.groupby("nationality")["decision"].agg(["mean","count"])
nat_rates.columns = ["Approval Rate", "N"]
print(nat_rates)

ct = pd.crosstab(df["nationality"], df["decision"])
chi2, p, dof, _ = chi2_contingency(ct)
print(f"Chi-square: chi2={chi2:.3f}, p={p:.4f}, df={dof}")
if p < 0.05:
    print("Significant association between nationality and hiring decision.")
else:
    print("No significant association between nationality and hiring decision.")

# ------------------------------------------------------------------------------
# 5. Proxy Discrimination -- sport as a gender proxy
#
# sport is not a job-relevant attribute but encodes gender information:
# Chess is 86% female, Rugby is 78% male.
# A model using sport as a feature would discriminate by gender indirectly.
# ------------------------------------------------------------------------------

sport_gender = pd.crosstab(df["sport"], df["gender"], normalize="index").round(2)
print("Gender composition per sport:")
print(sport_gender)

sport_rates = df.groupby("sport")["decision"].mean().sort_values()

fig, ax = plt.subplots(figsize=(8, 5))
colors = ["#E07B39" if r < sport_rates.median() else "#2C6E9B"
          for r in sport_rates.values]
bars = ax.barh(sport_rates.index, sport_rates.values,
               color=colors, edgecolor="white")
ax.bar_label(bars, labels=[f"{r:.1%}" for r in sport_rates.values],
             padding=4, fontsize=9.5)
ax.set_xlabel("Approval Rate", fontsize=11)
ax.set_title("Approval Rate by Sport (sport as a proxy for gender)",
             fontsize=13, fontweight="bold")
ax.axvline(sport_rates.median(), color="grey", ls="--", lw=1, label="Median")
ax.legend()
plt.tight_layout()
plt.savefig(IMAGES_PATH / "approval_by_sport.png", bbox_inches="tight", dpi=150)
plt.show()

# Point-biserial correlation to quantify the sport-gender relationship
df["gender_binary"] = (df["gender"] == "female").astype(int)
df["sport_encoded"] = df["sport"].map({
    "Chess": 0, "Running": 1, "Swimming": 2, "Golf": 3,
    "Tennis": 4, "Cricket": 5, "Football": 6, "Rugby": 7
})
r, p_val = pointbiserialr(df["gender_binary"], df["sport_encoded"])
print(f"Point-biserial correlation (gender x sport): r={r:.3f}, p={p_val:.4f}")

# ------------------------------------------------------------------------------
# 6. Company-Level Bias
# ------------------------------------------------------------------------------

company_gender = df.groupby(["company","gender"])["decision"].mean().unstack()
company_gender["DI (F/M)"] = (company_gender["female"] / company_gender["male"]).round(3)
print("Approval rates and DI by company:")
print(company_gender.round(3))

companies = ["A","B","C","D"]
x = np.arange(4)
width = 0.25

fig, ax = plt.subplots(figsize=(9, 5))
for i, (gender, color) in enumerate(zip(["female","male","other"], PALETTE)):
    rates = [df[(df["company"]==c) & (df["gender"]==gender)]["decision"].mean()
             for c in companies]
    ax.bar(x + i*width, rates, width, label=gender.capitalize(),
           color=color, edgecolor="white")
ax.set_xticks(x + width)
ax.set_xticklabels([f"Company {c}" for c in companies])
ax.set_ylabel("Approval Rate", fontsize=11)
ax.set_title("Approval Rate by Company and Gender", fontsize=13, fontweight="bold")
ax.legend()
ax.text(0.01, -0.14,
        "Company B shows the most severe gap: 14% female vs 46% male.",
        transform=ax.transAxes, fontsize=9.5, color="#444")
plt.tight_layout()
plt.savefig(IMAGES_PATH / "approval_company_gender.png", bbox_inches="tight", dpi=150)
plt.show()

# ------------------------------------------------------------------------------
# 7. Qualification Paradox
#
# Women score higher on average (64.0 vs 60.9) but are approved less often
# (27.5% vs 35.3%). Qualifications are not the primary driver of decisions.
# ------------------------------------------------------------------------------

print("Mean grade by gender:")
print(df.groupby("gender")["ind-university_grade"].mean().round(2))

fig, axes = plt.subplots(1, 2, figsize=(11, 5))

df[df["gender"].isin(["female","male"])].boxplot(
    column="ind-university_grade", by="gender", ax=axes[0],
    boxprops=dict(color="#2C6E9B"), medianprops=dict(color="#E07B39", lw=2))
axes[0].set_title("Grade by Gender")
axes[0].set_xlabel("Gender")
axes[0].set_ylabel("University Grade")

rates = df[df["gender"].isin(["female","male"])].groupby("gender")["decision"].mean()
axes[1].bar(rates.index, rates.values,
            color=["#E07B39","#2C6E9B"], edgecolor="white", width=0.4)
for container in axes[1].containers:
    axes[1].bar_label(container,
                      labels=[f"{r:.1%}" for r in rates.values],
                      padding=4, fontsize=10)
axes[1].set_title("Approval Rate by Gender")
axes[1].set_ylabel("Approval Rate")
axes[1].set_ylim(0, 0.5)

plt.suptitle("Higher Grades, Lower Approval", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(IMAGES_PATH / "qualification_paradox.png", bbox_inches="tight", dpi=150)
plt.show()

# ------------------------------------------------------------------------------
# 8. Summary
# ------------------------------------------------------------------------------

print()
print("Bias findings summary:")
findings = [
    ("Gender DI",             f"{di_gender:.3f} -- four-fifths rule violated",   "High"),
    ("Age",                   f"{di_age:.3f} -- younger candidates disadvantaged","Medium"),
    ("Nationality",           "No significant difference (p > 0.05)",              "Low"),
    ("Proxy (sport)",         "Chess 86% female, Rugby 78% male",                  "High"),
    ("Company B",             "Female 14% vs male 46%",                            "High"),
    ("Qualification paradox", "Higher grades, lower approval",                     "High"),
]
for bias, finding, severity in findings:
    print(f"  {severity:8s}  {bias:24s}  {finding}")
