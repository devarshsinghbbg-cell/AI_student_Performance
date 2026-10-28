"""Scores the System Usability Scale (Brooke, 1996). Input CSV: one row per participant, columns q1..q10 with answers 1-5.
Usage: python score_sus.py responses.csv   -> prints each participant's SUS (0-100) and the mean/sd."""
import sys, pandas as pd
d = pd.read_csv(sys.argv[1]); q = d[[f"q{i}" for i in range(1, 11)]]
odd = q[[f"q{i}" for i in (1,3,5,7,9)]] - 1; even = 5 - q[[f"q{i}" for i in (2,4,6,8,10)]]
d["SUS"] = (odd.sum(axis=1) + even.sum(axis=1)) * 2.5; print(d[["SUS"]].round(1).to_string()); print(f"n={len(d)}  mean={d.SUS.mean():.1f}  sd={d.SUS.std():.1f}")
