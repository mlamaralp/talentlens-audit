# TalentLens -- Data Quality Audit
# Task 1: assessing the recruitment dataset for data quality issues

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


# 2. Dataset Overview
print(df.dtypes)
print(df.describe().T.round(2))

# 3. Data Quality Assessment -- to be completed
# 4. Summary -- to be completed
