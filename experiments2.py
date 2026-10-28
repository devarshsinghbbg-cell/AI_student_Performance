"""Phase-2 experiments: tuning, imbalance handling, calibration, thresholds, exact SHAP comparison, fairness audit.
Run AFTER train_model.py. Overwrites outputs/web_model.json with the calibrated model + risk-band thresholds."""
import warnings; warnings.filterwarnings("ignore")
import json, itertools, math, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (roc_auc_score, average_precision_score, precision_score, recall_score, f1_score,
                             brier_score_loss, confusion_matrix)
SEED = 42; rng = np.random.default_rng(SEED)
df = pd.read_csv("data/students.csv")
F = ["attendance_pct","quiz_avg","assignment_avg","assignment_trend","coding_hours_week","study_hours_week","missed_deadlines"]
Xtr, Xte, ytr, yte = train_test_split(df[F], df.at_risk, test_size=0.2, stratify=df.at_risk, random_state=SEED)
skf = StratifiedKFold(5, shuffle=True, random_state=SEED); R = {}

def metrics(y, p, thr=0.5):
    pr = (p >= thr).astype(int)
    return dict(ROC_AUC=roc_auc_score(y,p), PR_AUC=average_precision_score(y,p), Precision=precision_score(y,pr,zero_division=0),
                Recall=recall_score(y,pr), F1=f1_score(y,pr))

# ---------- 1. Hyper-parameter tuning (5-fold CV, ROC-AUC, training set only) ----------
grids = {
 "Logistic Regression": (make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")), {"logisticregression__C":[0.01,0.1,1,10]}),
 "Random Forest": (RandomForestClassifier(class_weight="balanced", random_state=SEED), {"n_estimators":[200,400],"max_depth":[4,6,None],"min_samples_leaf":[1,5]}),
 "Gradient Boosting": (GradientBoostingClassifier(random_state=SEED), {"n_estimators":[100,200],"learning_rate":[0.05,0.1],"max_depth":[2,3]}),
}
tuned, rows = {}, []
for n,(est,g) in grids.items():
    gs = GridSearchCV(est, g, cv=skf, scoring="roc_auc", n_jobs=-1).fit(Xtr, ytr); tuned[n] = gs.best_estimator_
    m = metrics(yte, gs.best_estimator_.predict_proba(Xte)[:,1])
    rows.append(dict(Model=n, Best_params=str({k.split("__")[-1]:v for k,v in gs.best_params_.items()}), CV_AUC=gs.best_score_, **m))
tun = pd.DataFrame(rows); print(tun.round(3).to_string(index=False)); tun.round(3).to_csv("outputs/tuning.csv", index=False)
bestC = tuned["Logistic Regression"].named_steps["logisticregression"].C

# ---------- 2. Imbalance handling (LR, same C) ----------
def smote(X, y, k=5):
    Xm = X[y==1]; nneed = int((y==0).sum() - (y==1).sum())
    nn = NearestNeighbors(n_neighbors=k+1).fit(Xm); idx = nn.kneighbors(Xm, return_distance=False)[:,1:]
    i = rng.integers(0, len(Xm), nneed); j = idx[i, rng.integers(0, k, nneed)]; gap = rng.random((nneed,1))
    Xs = Xm[i] + gap*(Xm[j]-Xm[i]); return np.vstack([X, Xs]), np.concatenate([y, np.ones(nneed, int)])
def ros(X, y):
    mi = np.where(y==1)[0]; extra = rng.choice(mi, int((y==0).sum()-len(mi))); return np.vstack([X, X[extra]]), np.concatenate([y, y[extra]])
def fit_variant(name, X, y):
    sc = StandardScaler().fit(X); Z = sc.transform(X); cw = "balanced" if name=="Class weights" else None
    if name=="Random oversampling": Z, y = ros(Z, y)
    if name=="SMOTE": Z, y = smote(Z, y)
    m = LogisticRegression(C=bestC, max_iter=2000, class_weight=cw).fit(Z, y); return lambda Xn: m.predict_proba(sc.transform(Xn))[:,1]
rows = []
for v in ["No handling","Class weights","Random oversampling","SMOTE"]:
    cv = []
    for a,b in skf.split(Xtr, ytr):
        pr = fit_variant(v, Xtr.values[a], ytr.values[a]); p = pr(Xtr.values[b]); cv.append(metrics(ytr.values[b], p))
    cvm = pd.DataFrame(cv).mean(); pt = fit_variant(v, Xtr.values, ytr.values)(Xte.values)
    rows.append(dict(Strategy=v, CV_Recall=cvm.Recall, CV_F1=cvm.F1, CV_PR_AUC=cvm.PR_AUC, **{"Test_"+k:val for k,val in metrics(yte, pt).items()}))
imb = pd.DataFrame(rows); print(imb.round(3).to_string(index=False)); imb.round(3).to_csv("outputs/imbalance.csv", index=False)

# ---------- 3. Calibration (Platt scaling on cross-validated log-odds) ----------
lr = tuned["Logistic Regression"]; sc = lr.named_steps["standardscaler"]; clf = lr.named_steps["logisticregression"]
cv_logit = cross_val_predict(lr, Xtr, ytr, cv=skf, method="decision_function")
platt = LogisticRegression(C=1e6).fit(cv_logit.reshape(-1,1), ytr); a, b = float(platt.coef_[0][0]), float(platt.intercept_[0])
te_logit = lr.decision_function(Xte); p_raw = 1/(1+np.exp(-te_logit)); p_cal = 1/(1+np.exp(-(a*te_logit+b)))
def ece(y, p, bins=10):
    e = np.linspace(0,1,bins+1); t = 0
    for i in range(bins):
        m = (p>=e[i])&(p<(e[i+1] if i<bins-1 else 1.01))
        if m.any(): t += m.mean()*abs(p[m].mean()-y[m].mean())
    return t
cal = pd.DataFrame([dict(Model="Class-weighted LR (raw)", Brier=brier_score_loss(yte,p_raw), ECE=ece(yte.values,p_raw), ROC_AUC=roc_auc_score(yte,p_raw)),
                    dict(Model="Class-weighted LR + Platt scaling", Brier=brier_score_loss(yte,p_cal), ECE=ece(yte.values,p_cal), ROC_AUC=roc_auc_score(yte,p_cal))])
print(cal.round(4).to_string(index=False)); cal.round(4).to_csv("outputs/calibration.csv", index=False)
fig, ax = plt.subplots(figsize=(4.6,4))
for p, lab, c in [(p_raw,"Raw (class-weighted)","#e03131"),(p_cal,"After Platt scaling","#2f9e44")]:
    bins = np.linspace(0,1,9); xs, ys = [], []
    for i in range(8):
        m = (p>=bins[i])&(p<bins[i+1]+(1e-9 if i==7 else 0))
        if m.sum()>=5: xs.append(p[m].mean()); ys.append(yte.values[m].mean())
    ax.plot(xs, ys, "o-", label=lab, color=c)
ax.plot([0,1],[0,1],"k--",lw=.8,label="Perfect"); ax.set_xlabel("Predicted risk"); ax.set_ylabel("Observed at-risk fraction"); ax.set_title("Reliability diagram (test set)"); ax.legend(fontsize=7)
plt.tight_layout(); plt.savefig("figures/reliability.png", dpi=200); plt.close()

# ---------- 4. Threshold selection on cross-validated calibrated probabilities (training data only) ----------
p_cv = 1/(1+np.exp(-(a*cv_logit+b))); grid = np.round(np.arange(0.05,0.96,0.01),2); tr = []
for t in grid:
    pr = (p_cv>=t).astype(int); tr.append((t, recall_score(ytr,pr), precision_score(ytr,pr,zero_division=0), f1_score(ytr,pr)))
tr = pd.DataFrame(tr, columns=["t","recall","precision","f1"])
t_high = float(tr.loc[tr.f1.idxmax(),"t"]); t_low = float(tr[tr.recall>=0.95].t.max())
fig, ax = plt.subplots(figsize=(5,3.6)); ax.plot(tr.t,tr.recall,label="Recall"); ax.plot(tr.t,tr.precision,label="Precision"); ax.plot(tr.t,tr.f1,label="F1")
ax.axvline(t_high,ls="--",c="#c92a2a",lw=1); ax.axvline(t_low,ls="--",c="#b86e00",lw=1)
ax.text(t_high+.01,.02,f"High >= {t_high:.2f}",fontsize=7,color="#c92a2a",transform=ax.get_xaxis_transform()); ax.text(t_low+.01,.02,f"Medium >= {t_low:.2f}",fontsize=7,color="#b86e00",transform=ax.get_xaxis_transform())
ax.set_xlabel("Threshold on calibrated risk"); ax.set_title("Threshold selection (cross-validated, training data)"); ax.legend(fontsize=7); plt.tight_layout(); plt.savefig("figures/thresholds.png", dpi=200); plt.close()
band = np.where(p_cal>=t_high,"High",np.where(p_cal>=t_low,"Medium","Low"))
bt = pd.DataFrame({"Band":["Low","Medium","High"]}); bt["Students"]=[(band==x).sum() for x in bt.Band]; bt["At-risk"]=[int(yte.values[band==x].sum()) for x in bt.Band]
bt["At-risk rate"]=(bt["At-risk"]/bt.Students).round(3); print(bt.to_string(index=False)); bt.to_csv("outputs/bands_test.csv", index=False)
final_pred = (p_cal>=t_high).astype(int); fm = metrics(yte, p_cal, t_high); tn,fp,fn,tp = confusion_matrix(yte, final_pred).ravel(); print("final@t_high", fm, tn,fp,fn,tp)

# ---------- 5. Exact Shapley values for tuned GB vs exact linear contributions ----------
gb = tuned["Gradient Boosting"]; bg = Xtr.sample(50, random_state=SEED).values; inst = Xte.sample(100, random_state=SEED).values; n = len(F)
def shap_exact(x):
    subsets = list(itertools.product([0,1], repeat=n)); X_all = np.empty((len(subsets)*len(bg), n))
    for s,mask in enumerate(subsets):
        B = bg.copy(); B[:, np.array(mask,bool)] = x[np.array(mask,bool)]; X_all[s*len(bg):(s+1)*len(bg)] = B
    v = gb.decision_function(X_all).reshape(len(subsets), len(bg)).mean(1); val = {m:v[i] for i,m in enumerate(subsets)}; phi = np.zeros(n)
    for j in range(n):
        for m in subsets:
            if m[j]==0:
                k = sum(m); m2 = list(m); m2[j]=1; phi[j] += math.factorial(k)*math.factorial(n-k-1)/math.factorial(n)*(val[tuple(m2)]-val[m])
    return phi
phi_gb = np.array([shap_exact(x) for x in inst])
phi_lr = clf.coef_[0] * (inst - bg.mean(0)) / sc.scale_      # exact SHAP of a linear model (log-odds scale)
top1 = np.mean(np.argmax(phi_gb,1)==np.argmax(phi_lr,1))
top3 = np.mean([len(set(np.argsort(-a_)[:3]) & set(np.argsort(-b_)[:3]))/3 for a_,b_ in zip(phi_gb,phi_lr)])
sign = np.mean(np.sign(phi_gb)==np.sign(phi_lr)); rho_i = np.nanmean([spearmanr(a_,b_)[0] for a_,b_ in zip(phi_gb,phi_lr)])
g_gb, g_lr = np.abs(phi_gb).mean(0), np.abs(phi_lr).mean(0); rho_g = spearmanr(g_gb,g_lr)[0]
xr = dict(top1_agreement=top1, top3_overlap=top3, sign_agreement=sign, mean_spearman_per_student=rho_i, global_rank_spearman=rho_g)
print(xr)
o = np.argsort(g_lr); y_ = np.arange(n); fig, ax = plt.subplots(figsize=(5.6,3.6))
ax.barh(y_-.2, (g_gb/g_gb.sum())[o], .4, label="Gradient boosting (exact Shapley)", color="#e8a317"); ax.barh(y_+.2, (g_lr/g_lr.sum())[o], .4, label="Logistic regression (coefficients)", color="#3b5bdb")
ax.set_yticks(y_); ax.set_yticklabels(np.array(F)[o]); ax.set_xlabel("Share of mean |contribution|"); ax.set_title("Which features drive risk: two models compared"); ax.legend(fontsize=7, loc="lower right"); plt.tight_layout(); plt.savefig("figures/shap_compare.png", dpi=200); plt.close()

# ---------- 6. Fairness-audit demonstration (simulated group attribute, NOT a model input) ----------
att = Xte.attendance_pct.values; r2 = np.random.default_rng(7); p_day = 1/(1+np.exp((att-76)/9)); grp = np.where(r2.random(len(att))<p_day, "Day scholar", "Hostel resident")
fa = []
for g in ["Hostel resident","Day scholar"]:
    m = grp==g; y_g = yte.values[m]; pr = final_pred[m]; tn_,fp_,fn_,tp_ = confusion_matrix(y_g, pr, labels=[0,1]).ravel()
    fa.append(dict(Group=g, N=int(m.sum()), Base_rate=y_g.mean(), Flag_rate=pr.mean(), TPR=tp_/max(tp_+fn_,1), FPR=fp_/max(fp_+tn_,1), Precision=tp_/max(tp_+fp_,1)))
fa = pd.DataFrame(fa); print(fa.round(3).to_string(index=False)); fa.round(3).to_csv("outputs/fairness_demo.csv", index=False)
fair = dict(eq_opp_gap=abs(fa.TPR[0]-fa.TPR[1]), dem_parity_gap=abs(fa.Flag_rate[0]-fa.Flag_rate[1]), fpr_gap=abs(fa.FPR[0]-fa.FPR[1]))

# ---------- 7. Export calibrated model for the web / backend ----------
web = json.load(open("outputs/web_model.json"))
web.update(dict(coef=[float(a*c) for c in clf.coef_[0]], intercept=float(a*clf.intercept_[0]+b), mean=sc.mean_.tolist(), scale=sc.scale_.tolist(),
                features=F, t_low=t_low, t_high=t_high, platt_a=a, platt_b=b, C=float(bestC),
                feature_means_safe=df[df.at_risk==0][F].mean().round(2).to_dict()))
json.dump(web, open("outputs/web_model.json","w"), indent=1)
json.dump(dict(platt_a=a, platt_b=b, t_low=t_low, t_high=t_high, final_metrics_at_t_high=fm, confusion_at_t_high=dict(TN=int(tn),FP=int(fp),FN=int(fn),TP=int(tp)),
               xai=xr, fairness_gaps=fair, best_params={r["Model"]:r["Best_params"] for r in tun.to_dict("records")}, C=float(bestC)), open("outputs/exp2_results.json","w"), indent=1, default=float)
print("done", dict(t_low=t_low, t_high=t_high, a=a, b=b))
