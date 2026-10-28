"""
Generates a SYNTHETIC student dataset for prototyping EduMentor AI.
NOTE: This is simulated data (not real students). Replace data/students.csv with real
institutional data (or a public dataset such as OULAD / UCI Student Performance) when available.
"""
import numpy as np, pandas as pd

rng = np.random.default_rng(42)
N = 1500
SUBJECTS = ["Maths", "Programming", "DBMS", "Networks", "AI_Basics"]

ability = rng.normal(0, 1, N)          # latent academic ability
effort  = rng.normal(0, 1, N)          # latent effort / discipline

def clip(x, lo=0, hi=100): return np.clip(x, lo, hi)

attendance   = clip(76 + 9*effort + 3*ability + rng.normal(0, 6, N))
study_hours  = clip(9 + 3.2*effort + rng.normal(0, 2.2, N), 0, 30)
coding_hours = clip(4 + 1.8*effort + 1.2*ability + rng.normal(0, 1.6, N), 0, 20)
login_freq   = clip(14 + 4*effort + rng.normal(0, 3, N), 0, 40)
missed_deadlines = np.clip(np.round(2.2 - 1.1*effort - 0.4*ability + rng.normal(0, 1.1, N)), 0, 10)

quiz = {s: clip(62 + 13*ability + 5*effort + rng.normal(0, 10, N)) for s in SUBJECTS}
quiz_avg = np.mean([quiz[s] for s in SUBJECTS], axis=0)
assign_avg = clip(66 + 9*ability + 6*effort + rng.normal(0, 8, N))
assign_trend = 0.8*effort + 0.3*ability + rng.normal(0, 1.2, N)    # +ve = improving

final = clip(0.38*quiz_avg + 0.27*assign_avg + 0.15*attendance + 1.3*study_hours*0.5
             + 0.6*coding_hours + 2.0*assign_trend - 1.2*missed_deadlines - 10
             + rng.normal(0, 4.5, N))
cut = np.percentile(final, 24)         # ~24% at-risk
at_risk = (final < cut).astype(int)

df = pd.DataFrame({
    "attendance_pct": attendance.round(1), "quiz_avg": quiz_avg.round(1),
    "assignment_avg": assign_avg.round(1), "assignment_trend": assign_trend.round(2),
    "coding_hours_week": coding_hours.round(1), "study_hours_week": study_hours.round(1),
    "lms_logins_month": login_freq.round(0), "missed_deadlines": missed_deadlines.astype(int),
    **{f"quiz_{s}": quiz[s].round(1) for s in SUBJECTS},
    "final_score": final.round(1), "at_risk": at_risk,
})
df.to_csv("data/students.csv", index=False)
print(df.shape, "at-risk rate:", at_risk.mean().round(3))
print(df.describe().T[["mean","std","min","max"]].round(2))
