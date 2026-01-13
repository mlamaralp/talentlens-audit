# TalentLens -- Recruitment Fairness Audit

Audit of a simulated AI-powered recruitment system for data quality issues,
algorithmic bias, and GDPR/EU AI Act compliance gaps. The analysis covers
4,000 hiring decisions across four companies, with a focus on gender disparate
impact and proxy discrimination.

---

## Scenario

TalentLens is a fictitious HR-tech company that uses an automated screening
tool to shortlist job applicants. Following a regulatory inquiry about potential
discrimination in its hiring decisions, this audit investigates:

1. The quality of the underlying data
2. Bias patterns in historical shortlisting decisions
3. Privacy and governance gaps relative to GDPR and the EU AI Act

---

## Dataset

**Source:** [Utrecht Fairness Recruitment Dataset](https://www.kaggle.com/datasets/ictinstitute/utrecht-fairness-recruitment-dataset) on Kaggle

The Utrecht dataset was chosen over other recruitment fairness datasets because
it is based on real hiring decision patterns rather than synthetically generated
data. It contains genuine inconsistencies and bias patterns that make the audit
more meaningful, and it has been used in peer-reviewed fairness research (ICT
Institute, 2025/2026). A synthetically generated dataset with pre-defined bias
would not provide a realistic data quality exercise.

The dataset contains 4,000 applicant records evaluated by four companies (A, B,
C, D), with features including gender, age, nationality, university grade, degree
level, sport, and a binary hiring decision.

---

## Repository Structure

```
talentlens-audit/
├── data/
│   └── recruitmentdataset-2022-1_3.csv
├── src/
│   ├── 01_data_quality.py
│   ├── 02_bias_analysis.py
│   ├── 03_governance.py
│   └── fairness_utils.py
├── images/                        # generated on first run
├── requirements.txt
└── README.md
```

Running any script creates `images/` automatically and saves all plots there.

---

## Tools

| Area | Stack |
|---|---|
| Language | Python 3.10+ |
| Data manipulation | Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn |
| Statistical tests | SciPy |

---

## Methodology

### Dataset choice

The Utrecht dataset was selected over synthetic alternatives because it reflects
real hiring decision patterns and contains genuine data quality issues. A dataset
generated specifically for bias detection exercises would not provide a meaningful
quality audit, and the bias patterns would be too obvious to require real analysis.

### Task 1 -- Data Quality

Six dimensions are assessed: completeness, uniqueness, validity, accuracy,
consistency, and data minimisation. No imputation is applied -- issues are
documented and quantified as found.

Key finding: no missing values or duplicates, but the university grade range
(45-78) is unusually narrow and likely reflects undocumented normalisation. The
sport and nationality fields have no documented job-relevant purpose, which is a
data minimisation violation under GDPR Article 5(1)(c).

### Task 2 -- Bias Detection

The primary metric is the Disparate Impact (DI) ratio based on the four-fifths
rule (EEOC Uniform Guidelines, 1978):

```
DI = approval_rate(unprivileged) / approval_rate(privileged)
```

A ratio below 0.8 indicates potential discriminatory impact.

Gender was chosen as the primary protected attribute because it is the most
directly measurable in this dataset and the most legally relevant under EU
employment law. Age and nationality were analysed as secondary attributes. The
sport field was investigated as a proxy variable because sports participation
patterns differ significantly by gender in this dataset.

### Task 3 -- Governance

All PII fields were mapped to their relevant GDPR articles. Pseudonymisation
was demonstrated using SHA-256 hashing with a salt, which allows the data
controller to fulfil erasure requests (GDPR Art. 17) without breaking
referential integrity. The system was classified under the EU AI Act as
high-risk (Annex III, Point 4a -- AI used in recruitment).

---

## Key Findings

### Data Quality

| Dimension | Finding | Status |
|---|---|---|
| Completeness | No missing values | Pass |
| Uniqueness | No duplicate records | Pass |
| Validity | All values within expected ranges | Pass |
| Accuracy | Grade range 45-78 unusually narrow | Warning |
| Consistency | Grade distributions consistent across companies | Pass |
| Data minimisation | sport and nationality collected without justification | Fail |

### Bias Detection

| Bias Type | Finding | Severity |
|---|---|---|
| Gender DI ratio | 0.780 -- below 0.8 threshold | Critical |
| Proxy discrimination | Chess 86% female, Rugby 78% male | Critical |
| Company B gender gap | Female 14% vs male 46% | Critical |
| Age | Younger candidates approved less often | Medium |
| Nationality | No significant association (p > 0.05) | Low |
| Qualification paradox | Women score higher but are approved less | High |

The DI ratio of 0.780 falls below the four-fifths rule threshold. The sport
field acts as a gender proxy: Chess is 86% female while Rugby is 78% male.
Female candidates have a higher mean university grade (64.0 vs 60.9) but are
approved at a lower rate (27.5% vs 35.3%).

### Governance Gaps

| Issue | Priority | Reference |
|---|---|---|
| Gender DI below 0.8 | Critical | EU AI Act Annex III; GDPR Art. 22 |
| sport as gender proxy | Critical | GDPR Art. 5(1)(c); EU AI Act Art. 10 |
| No human oversight for rejections | High | GDPR Art. 22; EU AI Act Art. 14 |
| No audit trail | High | EU AI Act Art. 12 |
| sport and nationality not minimised | High | GDPR Art. 5(1)(c); Art. 25 |
| No lawful basis documented | High | GDPR Art. 6; Art. 30 |

---

## EU AI Act Classification

TalentLens is classified as a high-risk AI system under Annex III, Point 4(a)
of the EU AI Act (2024), which explicitly covers AI used for shortlisting job
applicants. This triggers mandatory obligations including conformity assessment,
human oversight, audit logging, and registration in the EU database of high-risk
AI systems before deployment.

---

## Governance Recommendations

1. Remove sport from the dataset and all downstream models -- no job-relevant
   justification and a confirmed gender proxy.
2. Conduct a full algorithmic audit and retrain any model using fairness
   constraints.
3. Implement mandatory human review for all rejection decisions (GDPR Art. 22).
4. Establish an audit trail logging decision timestamp, model version, input
   features, and reviewer ID for every record.
5. Commission a Data Protection Impact Assessment (DPIA) before further
   deployment.
6. Document a lawful basis under GDPR Art. 6 and create a Records of Processing
   Activities (ROPA) entry.

---

## How to Run

```bash
# 1. Clone the repository
git clone https://github.com/mlamaralp/talentlens-audit.git
cd talentlens-audit

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the dataset from Kaggle and place it in data/
# https://www.kaggle.com/datasets/ictinstitute/utrecht-fairness-recruitment-dataset

# 4. Run scripts in order
python src/01_data_quality.py
python src/02_bias_analysis.py
python src/03_governance.py
```

---

## References

- ICT Institute (2025). Utrecht Fairness Recruitment Dataset. Kaggle.
- Burda, P. & van Otterloo, S. (2026). Fairness trade-offs in hiring. ACM HCAI-ep.
- European Commission (2024). Regulation (EU) 2024/1689 -- EU AI Act.
- EEOC (1978). Uniform Guidelines on Employee Selection Procedures. 29 CFR Part 1607.
- Regulation (EU) 2016/679 -- General Data Protection Regulation (GDPR).

---

## License

MIT License.
