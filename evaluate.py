"""Score a detector run with and without VerdictAD.

Image AUROC / Image AUPR on s (detector) and S = s * sigmoid((z_free + z_ref) / 2) ** alpha;
Pixel AUROC / AUPRO on M (detector map) and M' = M * W, W = window weights with exponent gamma.  AUPRO is SubspaceAD's compute_aupro (FPR limit 0.3).

usage: python evaluate.py RUN_DIR DECISION_DIR (decide.py output, <cls>.pkl, or the released <cls>.json) --subspacead PATH [--alpha 0.5] [--gamma 0.25] [--windows merged|crop|box]"""
import argparse, glob, json, os, pickle, sys
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from verdictad import gate_score, apply_windows

p = argparse.ArgumentParser()
p.add_argument("run_dir"); p.add_argument("decision_dir"); p.add_argument("--subspacead", required=True)
p.add_argument("--alpha", type=float, default=0.5); p.add_argument("--gamma", type=float, default=0.25)
p.add_argument("--windows", default="merged", choices=["merged", "crop", "box"]); p.add_argument("--classes", default="")
a = p.parse_args()
sys.path.insert(0, a.subspacead); from main import compute_aupro
aupro = lambda m, g: compute_aupro(list(m), list(g), fpr_limit=0.3, num_thresholds=300, connectivity=8)
key = lambda q: "/".join(str(q).replace("\\", "/").split("/")[-4:])
win = {"merged": slice(0, 32), "crop": slice(0, 16), "box": slice(16, 32)}[a.windows]

load = lambda c: json.load(open(f"{a.decision_dir}/{c}.json")) if os.path.exists(f"{a.decision_dir}/{c}.json") else pickle.load(open(f"{a.decision_dir}/{c}.pkl", "rb"))
cls_list = [c for c in a.classes.split(",") if c] or sorted(os.path.splitext(os.path.basename(f))[0] for f in glob.glob(a.decision_dir + "/*.pkl") + glob.glob(a.decision_dir + "/*.json"))
res = {k: [] for k in ("ia0", "ip0", "pa0", "pro0", "ia1", "ip1", "pa1", "pro1")}
for c in cls_list:
    d = json.load(open(f"{a.run_dir}/imgscores_{c}.json")); y = np.array(d["img_true"]); s = np.array(d["img_pred"], dtype=np.float64)
    rows = {key(r["img_path"]): r for r in load(c)}; R = [rows[key(q)] for q in d["paths"]]
    z = np.load(f"{a.run_dir}/imgmaps_{c}.npz"); M = z["maps"].astype(np.float32); G = z["gts"]
    S = gate_score(s, np.array([0.5 * (r["z_free"] + r["z_ref"]) for r in R]), a.alpha)
    M1 = np.stack([apply_windows(m, r["boxes"][win], np.asarray(r["logits"])[win], a.gamma) for m, r in zip(M, R)])
    out = dict(ia0=roc_auc_score(y, s), ip0=average_precision_score(y, s), pa0=roc_auc_score(G.ravel(), M.ravel()), pro0=aupro(M, G),
               ia1=roc_auc_score(y, S), ip1=average_precision_score(y, S), pa1=roc_auc_score(G.ravel(), M1.ravel()), pro1=aupro(M1, G))
    for k, v in out.items(): res[k].append(float(v))
    print(c, " / ".join(f"{100 * out[k]:.2f}" for k in ("ia0", "ip0", "pa0", "pro0")), "->", " / ".join(f"{100 * out[k]:.2f}" for k in ("ia1", "ip1", "pa1", "pro1")), flush=True)
print(f"MEAN over {len(cls_list)} classes (Image AUROC / Image AUPR / Pixel AUROC / AUPRO)")
print("  detector              ", " / ".join(f"{100 * np.mean(res[k]):.2f}" for k in ("ia0", "ip0", "pa0", "pro0")))
print("  + VerdictAD           ", " / ".join(f"{100 * np.mean(res[k]):.2f}" for k in ("ia1", "ip1", "pa1", "pro1")))
