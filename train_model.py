"""Trains and evaluates academic-risk models; exports figures, metrics and web-model JSON."""
import json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay)
from sklearn.inspection import permutation_importance

SEED = 42
df = pd.read_csv("data/students.csv")
FEATURES = ["attendance_pct","quiz_avg","assignment_avg","assignment_trend",
            "coding_hours_week","study_hours_week","missed_deadlines"]
# NOTE: lms_logins_month was tried in v1 but dropped: it received a small *positive* risk coefficient
# (collinearity with study/coding effort), which made explanations misleading.
X, y = df[FEATURES], df["at_risk"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

models = {
 "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")),
 "Decision Tree": DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=SEED),
 "Random Forest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=SEED),
 "Gradient Boosting": GradientBoostingClassifier(random_state=SEED),
}
skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
rows, probs = [], {}
for name, m in models.items():
    cv = cross_val_score(m, Xtr, ytr, cv=skf, scoring="roc_auc")
    m.fit(Xtr, ytr); p = m.predict_proba(Xte)[:,1]; pred = (p >= 0.5).astype(int); probs[name] = p
    rows.append(dict(Model=name, Accuracy=accuracy_score(yte,pred), Precision=precision_score(yte,pred),
        Recall=recall_score(yte,pred), F1=f1_score(yte,pred), ROC_AUC=roc_auc_score(yte,p),
        CV_AUC_mean=cv.mean(), CV_AUC_std=cv.std()))
res = pd.DataFrame(rows).round(3); print(res.to_string(index=False))
res.to_csv("outputs/model_comparison.csv", index=False)

# ---- Ablation: which feature groups matter (Logistic Regression) ----
groups = {
 "Attendance only": ["attendance_pct"],
 "Scores only (quiz+assignment)": ["quiz_avg","assignment_avg"],
 "Behaviour only (study/coding/deadlines/trend)": ["assignment_trend","coding_hours_week","study_hours_week","missed_deadlines"],
 "All features (EduMentor)": FEATURES,
}
abl = []
for g, cols in groups.items():
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")).fit(Xtr[cols], ytr)
    p = m.predict_proba(Xte[cols])[:,1]; pr = (p>=.5).astype(int)
    abl.append(dict(Feature_set=g, F1=f1_score(yte,pr), Recall=recall_score(yte,pr), ROC_AUC=roc_auc_score(yte,p)))
abl = pd.DataFrame(abl).round(3); print(abl.to_string(index=False)); abl.to_csv("outputs/ablation.csv", index=False)

# ---- Figures ----
plt.figure(figsize=(5.2,4))
for n,p in probs.items():
    fpr,tpr,_ = roc_curve(yte,p); plt.plot(fpr,tpr,label=f"{n} (AUC={roc_auc_score(yte,p):.3f})")
plt.plot([0,1],[0,1],"k--",lw=.8); plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
plt.title("ROC curves – academic risk prediction"); plt.legend(fontsize=7); plt.tight_layout()
plt.savefig("figures/roc.png", dpi=200); plt.close()

best_name = res.sort_values("ROC_AUC", ascending=False).iloc[0]["Model"]
ConfusionMatrixDisplay(confusion_matrix(yte,(probs[best_name]>=.5).astype(int)),
    display_labels=["Safe","At-risk"]).plot(cmap="Blues", colorbar=False)
plt.title(f"Confusion matrix – {best_name}"); plt.tight_layout(); plt.savefig("figures/confusion.png", dpi=200); plt.close()

# permutation importance on the Logistic Regression model
lr = models["Logistic Regression"]
pi = permutation_importance(lr, Xte, yte, scoring="roc_auc", n_repeats=20, random_state=SEED)
order = np.argsort(pi.importances_mean)
plt.figure(figsize=(5.4,3.8)); plt.barh(np.array(FEATURES)[order], pi.importances_mean[order], xerr=pi.importances_std[order], color="#3b5bdb")
plt.xlabel("Drop in ROC-AUC when feature shuffled"); plt.title("Permutation feature importance"); plt.tight_layout()
plt.savefig("figures/importance.png", dpi=200); plt.close()

fig,ax = plt.subplots(1,2,figsize=(7,3)); 
df["at_risk"].map({0:"Safe",1:"At-risk"}).value_counts().plot.bar(ax=ax[0], color=["#2f9e44","#e03131"]); ax[0].set_title("Class distribution"); ax[0].tick_params(axis="x", rotation=0)
corr = df[FEATURES+["final_score"]].corr()["final_score"].drop("final_score").sort_values()
corr.plot.barh(ax=ax[1], color="#3b5bdb"); ax[1].set_title("Correlation with final score"); plt.tight_layout()
plt.savefig("figures/eda.png", dpi=200); plt.close()

# ---- Export LR for the web prototype ----
sc, clf = lr.named_steps["standardscaler"], lr.named_steps["logisticregression"]
web = dict(features=FEATURES, mean=sc.mean_.tolist(), scale=sc.scale_.tolist(),
           coef=clf.coef_[0].tolist(), intercept=float(clf.intercept_[0]),
           feature_means_safe=df[df.at_risk==0][FEATURES].mean().round(2).to_dict())
json.dump(web, open("outputs/web_model.json","w"), indent=1)
json.dump(dict(best_model=best_name, n_train=len(Xtr), n_test=len(Xte), at_risk_rate=float(y.mean()),
     lr_coef=dict(zip(FEATURES, clf.coef_[0].round(3).tolist()))), open("outputs/summary.json","w"), indent=1)
print("best:", best_name); print("LR coefs:", dict(zip(FEATURES, clf.coef_[0].round(2))))
