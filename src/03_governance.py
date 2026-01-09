# TalentLens -- Privacy and Governance Assessment
# Task 3: GDPR compliance, PII handling, and governance recommendations
#
# Dataset: Utrecht Fairness Recruitment Dataset
# Goal: identify PII, demonstrate pseudonymisation, map findings to
# GDPR and EU AI Act, and propose concrete governance controls.

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


# ------------------------------------------------------------------------------
# 2. PII Identification
# ------------------------------------------------------------------------------

pii = [
    ("Id",                  "Direct identifier",                          "High",   "Art. 4(1)"),
    ("gender",              "Sensitive attribute (Art. 9 adjacent)",      "High",   "Art. 9"),
    ("age",                 "Sensitive attribute (discrimination risk)",  "High",   "Directive 2000/78/EC"),
    ("nationality",         "Sensitive attribute",                        "High",   "Art. 9"),
    ("sport",               "Indirect identifier -- gender proxy",        "High",   "Art. 22"),
    ("ind-university_grade","Personal data",                              "Medium", "Art. 5(1)(c)"),
    ("ind-debateclub",      "Personal data",                              "Low",    "Art. 5(1)(c)"),
    ("ind-programming_exp", "Personal data",                              "Low",    "Art. 5(1)(c)"),
    ("ind-international_exp","Personal data",                             "Low",    "Art. 5(1)(c)"),
    ("ind-entrepeneur_exp", "Personal data",                              "Low",    "Art. 5(1)(c)"),
    ("ind-languages",       "Personal data",                              "Low",    "Art. 5(1)(c)"),
    ("ind-exact_study",     "Personal data",                              "Low",    "Art. 5(1)(c)"),
    ("ind-degree",          "Personal data",                              "Low",    "Art. 5(1)(c)"),
    ("company",             "Organisational data",                        "Low",    "N/A"),
    ("decision",            "Automated decision output",                  "High",   "Art. 22"),
]

print(f"{'Field':<25} {'PII Type':<45} {'Risk':<8} {'GDPR'}")
print("-" * 100)
for field, pii_type, risk, gdpr in pii:
    print(f"{field:<25} {pii_type:<45} {risk:<8} {gdpr}")

# ------------------------------------------------------------------------------
# 3. Pseudonymisation Demonstration
#
# The Id field is a direct identifier. Under GDPR Article 25 (data protection
# by design), identifiers should be pseudonymised before analysis.
# SHA-256 hashing with a salt is used here. The result cannot be reversed
# without the salt, which the data controller retains to handle erasure
# requests under GDPR Article 17.
# ------------------------------------------------------------------------------

SALT = "talentlens_2025_audit"

def pseudonymise(value: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}{value}".encode()).hexdigest()[:16]

df_pseudo = df.copy()
df_pseudo["Id_original"] = df_pseudo["Id"]
df_pseudo["Id"] = df_pseudo["Id"].apply(lambda x: pseudonymise(x, SALT))

print("\nOriginal vs pseudonymised IDs (first 5 rows):")
print(df_pseudo[["Id_original", "Id"]].head().to_string(index=False))
print("\nThe original ID cannot be recovered from the hash without the salt.")

# ------------------------------------------------------------------------------
# 4. Data Minimisation
#
# Fields with no documented job-relevant justification should be removed
# before sharing the dataset externally (GDPR Art. 5(1)(c)).
# ------------------------------------------------------------------------------

fields_to_remove = ["sport", "nationality"]
print("\nFields recommended for removal:")
print("  - sport: no job-relevant purpose; acts as a gender proxy")
print("  - nationality: no job-relevant purpose; discrimination risk")

keep_cols = [c for c in df.columns if c not in fields_to_remove]
print("\nRemaining fields after minimisation:")
print(keep_cols)

# ------------------------------------------------------------------------------
# 5. EU AI Act Classification
#
# TalentLens is HIGH-RISK under Annex III, Point 4(a) of the EU AI Act (2024).
# AI systems used for shortlisting job applicants are explicitly listed.
# ------------------------------------------------------------------------------

print("\nEU AI Act -- Risk Classification")
print("Article: Annex III, Point 4(a)")
print("Classification: HIGH-RISK")
print()
print("Mandatory obligations:")
for i, o in enumerate([
    "Conformity assessment before deployment",
    "Human oversight -- no fully automated final decisions",
    "Audit trail for all decisions",
    "Data governance documentation",
    "Transparency to applicants (Art. 13)",
    "Registration in EU database of high-risk AI systems",
    "Post-market monitoring plan"
], 1):
    print(f"  {i}. {o}")

# ------------------------------------------------------------------------------
# 6. Governance Recommendations
# ------------------------------------------------------------------------------

recs = [
    ("Gender DI = 0.780",                   "Full algorithmic audit; retrain with fairness constraints", "Critical", "EU AI Act Annex III; GDPR Art. 22"),
    ("sport as gender proxy",               "Remove sport from dataset and models immediately",          "Critical", "GDPR Art. 5(1)(c); EU AI Act Art. 10"),
    ("No human oversight",                  "Mandatory human review for all rejections",                "High",     "GDPR Art. 22; EU AI Act Art. 14"),
    ("No audit trail",                      "Log timestamp, model version, features per decision",      "High",     "EU AI Act Art. 12"),
    ("Age bias",                            "Audit whether age correlates with a genuine job requirement","Medium",  "Directive 2000/78/EC"),
    ("Data minimisation violations",        "Remove nationality and sport; document retained fields",   "High",     "GDPR Art. 5(1)(c); Art. 25"),
    ("No lawful basis documented",          "Establish basis under Art. 6 and create a ROPA entry",    "High",     "GDPR Art. 6; Art. 30"),
]

print()
print(f"{'Issue':<36} {'Priority':<10} {'Reference'}")
print("-" * 90)
for issue, rec, priority, ref in recs:
    print(f"{issue:<36} {priority:<10} {ref}")
    print(f"  Recommendation: {rec}")
    print()

# ------------------------------------------------------------------------------
# 7. GDPR Article Mapping
# ------------------------------------------------------------------------------

gdpr = [
    ("Art. 4(1)",    "Personal data definition",      "All applicant records are personal data"),
    ("Art. 5(1)(c)", "Data minimisation",             "sport and nationality lack documented necessity"),
    ("Art. 6",       "Lawful basis",                  "Recruitment requires explicit lawful basis -- not documented"),
    ("Art. 9",       "Special category data",         "Gender and nationality require extra protection"),
    ("Art. 13",      "Right to information",          "Applicants must be told automated screening is used"),
    ("Art. 17",      "Right to erasure",              "Salt-based pseudonymisation enables erasure without data loss"),
    ("Art. 22",      "Automated decisions",           "Applicants have the right to human review of rejections"),
    ("Art. 25",      "Data protection by design",     "Sensitive fields should be excluded at schema level"),
    ("Art. 30",      "Records of processing",         "Purpose, retention period and legal basis must be documented"),
]

print(f"{'Article':<14} {'Title':<32} {'Relevance'}")
print("-" * 90)
for article, title, relevance in gdpr:
    print(f"{article:<14} {title:<32} {relevance}")

# ------------------------------------------------------------------------------
# 8. Conclusion
# ------------------------------------------------------------------------------

print()
print("Governance gaps identified:")
print()
gaps = {
    "Critical": [
        "Gender DI = 0.780 -- four-fifths rule violated",
        "sport encodes gender and has no job-relevant purpose"
    ],
    "High": [
        "No audit trail for hiring decisions",
        "No documented human oversight mechanism",
        "sport and nationality collected without minimisation justification",
        "No lawful basis or ROPA entry"
    ],
    "Medium": [
        "Age bias -- younger candidates approved less often",
        "Grade range 45-78 unusually narrow"
    ]
}
for severity, items in gaps.items():
    print(f"{severity}:")
    for item in items:
        print(f"  - {item}")
    print()
print("Recommended next step: commission a DPIA before further deployment.")
