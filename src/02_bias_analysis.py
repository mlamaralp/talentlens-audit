# TalentLens -- Bias Detection and Fairness Analysis
# Task 2: detecting algorithmic bias in hiring decisions

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


# 2. Gender Disparate Impact
gender_rates = df.groupby("gender")["decision"].agg(["mean","count"]).round(3)
gender_rates.columns = ["Approval Rate", "N"]
print(gender_rates)

female_rate = df[df["gender"]=="female"]["decision"].mean()
male_rate   = df[df["gender"]=="male"]["decision"].mean()
di_gender   = female_rate / male_rate
print(f"DI ratio (female / male): {di_gender:.3f}")

fig, ax = plt.subplots(figsize=(7, 5))
genders = ["female","male","other"]
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
plt.tight_layout()
plt.savefig(IMAGES_PATH / "approval_by_gender.png", bbox_inches="tight", dpi=150)
plt.show()

# 3. Age Bias -- to be completed
# 4. Nationality -- to be completed
# 5. Proxy Discrimination -- to be completed
# 6. Company-Level Bias -- to be completed
# 7. Qualification Paradox -- to be completed
# 8. Summary -- to be completed
