"""Validation on REAL data: UCI 'Student Performance' (Cortez & Silva, 2008; CC BY 4.0).
Usage:
    python validate_uci.py path/to/student-mat.csv        # CSV from the UCI zip (semicolon-separated)
    python validate_uci.py                                 # or: pip install ucimlrepo (downloads the data)
Defines at-risk as final grade G3 < 10 (fail on the 0-20 Portuguese scale).
Two settings: EARLY (no G1/G2 - what an early-warning system actually has) and WITH_G1G2 (upper bound).
Writes outputs/uci_results.csv and prints a table to paste into the report (Section 7.12)."""
import sys, numpy as np, pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

if len(sys.argv) > 1: df = pd.read_csv(sys.argv[1], sep=";")
else:
    from ucimlrepo import fetch_ucirepo; d = fetch_ucirepo(id=320); df = pd.concat([d.data.features, d.data.targets], axis=1)
    df = df[~df.index.duplicated()]
assert "G3" in df.columns, "G3 column not found"
y = (df["G3"] < 10).astype(int); print(f"{len(df)} students, at-risk (G3<10): {y.mean():.1%}")
BEHAV = ["studytime","failures","absences","schoolsup","famsup","paid","activities","higher","internet","freetime","goout","Dalc","Walc","health","famrel","traveltime"]
SETTINGS = {"EARLY (no G1/G2)": BEHAV, "WITH_G1G2": BEHAV + ["G1","G2"]}
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42); rows = []
for sname, cols in SETTINGS.items():
    X = df[cols]; cat = [c for c in cols if not pd.api.types.is_numeric_dtype(X[c])]; num = [c for c in cols if c not in cat]
    pre = ColumnTransformer([("n", StandardScaler(), num), ("c", OneHotEncoder(handle_unknown="ignore"), cat)])
    for mname, m in {"Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
                     "Random Forest": RandomForestClassifier(300, class_weight="balanced", random_state=42),
                     "Gradient Boosting": GradientBoostingClassifier(random_state=42)}.items():
        r = cross_validate(make_pipeline(pre, m), X, y, cv=cv, scoring=["roc_auc","average_precision","recall","precision","f1"], n_jobs=-1)
        rows.append(dict(Setting=sname, Model=mname, **{k: f"{r['test_'+k].mean():.3f} +/- {r['test_'+k].std():.3f}" for k in ["roc_auc","average_precision","recall","precision","f1"]}))
out = pd.DataFrame(rows); out.to_csv("outputs/uci_results.csv", index=False); print(out.to_string(index=False))
