"""AnomalyDINO full-resolution anomaly maps (official run, tiff) x Jev dense-window weights, scored with AnomalyDINO's own eval_segmentation
(PRO via compute_pro, pixel AUROC via roc_auc_score on all pixels).  M' = M * W, W = sigmoid(window logit)^gamma, overlapping boxes averaged."""
import sys, os, pickle, argparse, numpy as np
from multiprocessing import Pool
from PIL import Image
p = argparse.ArgumentParser()
p.add_argument("--repo", required=True); p.add_argument("--data_root", required=True); p.add_argument("--maps_dir", required=True)
p.add_argument("--region_dir", required=True); p.add_argument("--classes", required=True); p.add_argument("--gamma", type=float, default=0.25)
p.add_argument("--procs", type=int, default=5); p.add_argument("--kind", default="MVTec")
a = p.parse_args(); sys.path.insert(0, a.repo)
from src.post_eval import parse_dataset_files, read_tiff, compute_pro, trapezoid
import glob, cv2
from src.utils import dists2map
from sklearn.metrics import roc_auc_score

def sig(x): return 1 / (1 + np.exp(-x))

def weight(boxes, logits, shape, g):
    H, W = shape; num = np.zeros(shape, np.float32); den = np.zeros(shape, np.float32); ws = sig(np.asarray(logits)) ** g
    for (x0, y0, x1, y1), w in zip(boxes, ws):
        num[int(y0 * H):int(round(y1 * H)), int(x0 * W):int(round(x1 * W))] += w; den[int(y0 * H):int(round(y1 * H)), int(x0 * W):int(round(x1 * W))] += 1
    return num / np.maximum(den, 1)

def load_map(pr):
    # full-resolution map: the stored tiff, or (when only the patch distances were stored) the same dists2map of the official code
    if not os.environ.get("FORCE_NPY"):
        try: return read_tiff(pr)
        except FileNotFoundError: pass
    q = pr.replace("\\", "/").split("/"); i = q.index("anomaly_maps"); rel = "/".join(q[i + 2:])   # <object>/test/<type>/<stem>
    img = glob.glob(os.path.join(a.data_root, rel) + ".*")[0]; H, W = Image.open(img).size[1], Image.open(img).size[0]
    return dists2map(np.load(pr + ".npy"), (H, W, 3))

def one(cls):
    gts, preds = parse_dataset_files(cls, a.data_root, a.maps_dir, a.kind)
    def k2(path):
        q = path.replace("\\", "/").split("/"); d = {"Anomaly": "bad", "Normal": "good"}.get(q[-2], q[-2]); return (d, os.path.splitext(q[-1])[0])
    reg = {k2(r["img_path"]): r for r in pickle.load(open(os.path.join(a.region_dir, cls + ".pkl"), "rb"))}
    G, P = [], []
    for g, pr in zip(gts, preds):
        m = load_map(pr); r = reg[k2(pr)]
        P.append(m * weight(r["boxes"], r["logits"], m.shape, a.gamma) if a.gamma > 0 else m)
        G.append(np.asarray(Image.open(g)) if g is not None else np.zeros(m.shape))
    curve = compute_pro(anomaly_maps=P, ground_truth_maps=G)
    pro = trapezoid(curve[0], curve[1], x_max=0.3) / 0.3
    auroc = roc_auc_score((np.array(G).ravel() > 0).astype(np.uint8), np.array(P).ravel())
    print(cls, round(auroc * 100, 2), round(pro * 100, 2), flush=True)
    return auroc, pro

if __name__ == "__main__":
    cl = a.classes.split(",")
    if a.procs == 1: res = [one(c) for c in cl]
    else:
        with Pool(min(a.procs, len(cl))) as pool: res = pool.map(one, cl)
    print(f"MEAN over {len(res)} classes gamma {a.gamma}: pixel AUROC {np.mean([r[0] for r in res])*100:.2f}  AUPRO {np.mean([r[1] for r in res])*100:.2f}")
