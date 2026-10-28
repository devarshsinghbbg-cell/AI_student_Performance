"""Injects the trained model + a sample cohort into web/template.html -> web/index.html"""
import json, pandas as pd
df = pd.read_csv("data/students.csv")
cols = ["attendance_pct","quiz_avg","assignment_avg","assignment_trend","coding_hours_week","study_hours_week",
        "lms_logins_month","missed_deadlines"] + [c for c in df.columns if c.startswith("quiz_") and c != "quiz_avg"]
cohort = df.sample(36, random_state=7)[cols].reset_index(drop=True)
model = json.load(open("outputs/web_model.json"))
t = open("web/template.html").read()
t = t.replace("__MODEL__", json.dumps(model)).replace("__COHORT__", json.dumps(cohort.to_dict("records")))
open("web/index.html", "w").write(t); print("built web/index.html", len(t), "bytes")
