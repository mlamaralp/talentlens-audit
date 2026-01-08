# TalentLens -- Privacy and Governance Assessment
# Task 3: GDPR compliance, PII handling, and governance recommendations

import os
import hashlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT        = Path(__file__).resolve().parent.parent
DATA_PATH   = ROOT / "data" / "recruitmentdataset-2022-1_3.csv"
IMAGES_PATH = ROOT / "images"
IMAGES_PATH.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({"figure.dpi": 130})

df = pd.read_csv(DATA_PATH)
print(f"Dataset: {df.shape[0]:,} rows x {df.shape[1]} columns")


# 2. PII Identification
pii = [
    ("Id",       "Direct identifier",                 "High",   "Art. 4(1)"),
    ("gender",   "Sensitive attribute",               "High",   "Art. 9"),
    ("age",      "Sensitive attribute",               "High",   "Directive 2000/78/EC"),
    ("sport",    "Indirect identifier -- gender proxy","High",  "Art. 22"),
    ("decision", "Automated decision output",         "High",   "Art. 22"),
]
print(f"{'Field':<12} {'Risk':<8} {'GDPR'}")
for field, _, risk, gdpr in pii:
    print(f"  {field:<12} {risk:<8} {gdpr}")

# 3. Pseudonymisation -- to be completed
# 4. Data Minimisation -- to be completed
# 5. EU AI Act -- to be completed
# 6. Recommendations -- to be completed
# 7. GDPR Mapping -- to be completed
# 8. Conclusion -- to be completed
