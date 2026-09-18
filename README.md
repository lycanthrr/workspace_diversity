# Workspace diversity & compensation analysis

Analysis of a ~10,000-employee dataset (org hierarchy + compensation data) to classify job levels, model salary, and assess pay fairness by department and gender.

This project works through four connected questions using an organizational hierarchy dataset and an employee compensation dataset:

1. **Job-level classification** — assign each employee to one of six levels (IC, Manager, Director, VP, Executive, CEO) based on their position in the reporting hierarchy.
2. **Span of control** — compute how many people each employee manages, both directly and indirectly (their full downline).
3. **Salary modeling** — train a Random Forest regressor to predict salary from department, experience, degree level, and gender, and examine which features actually drive pay.
4. **Fairness analysis** — determine whether gender predicts salary directly, or whether any pay gap is better explained by department composition.

##

- `company_hierarchy.csv` — `employee_id`, `boss_id`, `dept`
- `employee.csv` — `employee_id`, `signing_bonus`, `salary`, `degree_level`, `sex`, `yrs_experience`
- ~10,000 employees across HR, Engineering, Marketing, and Sales

## Methodology

- **Level classification**: derived each employee's level from their position in the boss/report chain (CASE WHEN–style mapping from hierarchy depth to IC → MM → D → VP → E → CEO).
- **Reports managed**: recursive traversal of the hierarchy to count direct and indirect reports per manager.
- **Salary model**: `RandomForestRegressor` (scikit-learn), evaluated with MSE, RMSE, explained variance, and prediction-within-25%-accuracy.
- **Feature importance & partial dependence**: used `feature_importances_` and `PartialDependenceDisplay` / `partial_dependence` to see which features move predicted salary, and how.

## Key Findings

- **Department is the dominant driver of salary** (feature importance 0.57), far ahead of years of experience (0.11) and being in Engineering (0.09). Being male (`is_male`) was only 0.03 — the weakest predictor in the model.
- **No direct pay discrimination by gender**: holding other factors constant, the model's predicted salary is close to flat across gender.
- **An indirect, structural pay gap exists**: HR — the lowest-paid department in the model — is 62.5% female, while Engineering — the highest-paid — is only 24.9% female. Because department drives pay so strongly, gender is linked to pay *through* which departments men and women are concentrated in, not through direct wage discrimination.

## Recommendations

- Don't treat this as "no gender problem" — the representation gap across departments *is* the problem, and it shows up in compensation outcomes.
- Investigate hiring pipelines, internal mobility, and promotion patterns into higher-paying departments (especially Engineering) to understand why representation skews the way it does.
- Track department-level gender composition over time as a leading indicator, not just aggregate pay-gap statistics.

## Tech Stack

Python, pandas, scikit-learn (RandomForestRegressor, PartialDependenceDisplay), matplotlib/seaborn

## Repository Structure

```
├── data/                      # company_hierarchy.csv, employee.csv
├── workplace_diversity_analysis.ipynb   # main analysis notebook
└── README.md
```

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook workplace_diversity_analysis.ipynb
```

---