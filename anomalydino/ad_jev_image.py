"""Image-level gate on AnomalyDINO's official anomaly maps (npy, meantop1p aggregation as in its eval_classification):
S = score * sigmoid(mean of unconditioned and reference-conditioned Noul logit)^e.  Image AUROC / AUPR (sklearn), mean over classes."""
import sys, os, pickle, argparse, numpy as np
from PIL import Image
p = argparse.ArgumentParser()
p.add_argument("--repo", required=True); p.add_argument("--dataset", required=True)  # MVTec | VisA | LOCO
p.add_argument("--data_root", required=True); p.add_argument("--maps_dir", required=True)
p.add_argument("--logit_u", required=True); p.add_argument("--logit_r", required=True); p.add_argument("--exps", default="0,0.5")
a = p.parse_args(); sys.path.insert(0, a.repo)
from src.post_eval import parse_dataset_files, mean_top1p, get_objects_from_dataset
from sklearn.metrics import roc_auc_score, average_precision_score
def sig(x): return 1 / (1 + np.exp(-x))
OBJ = {"LOCO": ['breakfast_box','juice_bottle','pushpins','screw_bag','splicing_connectors']}
objs = OBJ.get(a.dataset) or get_objects_from_dataset(a.dataset)
def lkey(path):
    q = path.replace("\\", "/").split("/"); stem = os.path.splitext(q[-1])[0]
    if a.dataset == "LOCO": return stem
    if a.dataset == "MVTec": return (q[-4], q[-2], stem)
    return (q[0], "good" if q[-2] == "Normal" else "bad", stem)
def pkey(path):
    q = path.split("/"); stem = q[-1]
    if a.dataset == "LOCO": return stem
    return (q[-4], q[-2], stem)
E_ = [float(x) for x in a.exps.split(",")]; res = {e: [] for e in E_}
for obj in objs:
    gts, preds = parse_dataset_files(obj, a.data_root, a.maps_dir, "MVTec" if a.dataset != "VisA" else "VisA")
    y = np.array([int(g is not None) for g in gts]); sc = np.array([mean_top1p(np.load(pn + ".npy").flatten()) for pn in preds])
    lu = {lkey(r["img_path"]): r["logit"] for r in pickle.load(open(os.path.join(a.logit_u, obj + ".pkl"), "rb"))}
    lr = {lkey(r["img_path"]): r["logit"] for r in pickle.load(open(os.path.join(a.logit_r, obj + ".pkl"), "rb"))}
    l = np.array([0.5 * (lu[pkey(pn)] + lr[pkey(pn)]) for pn in preds])
    for e in E_:
        s = sc * sig(l) ** e; res[e].append((roc_auc_score(y, s), average_precision_score(y, s)))
for e in E_:
    print(f"{a.dataset} exponent {e}: Image AUROC {np.mean([x[0] for x in res[e]])*100:.2f}  Image AUPR {np.mean([x[1] for x in res[e]])*100:.2f}  ({len(res[e])} classes)")
