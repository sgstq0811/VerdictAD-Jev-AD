"""AU-sPRO on the official MVTec LOCO AD release, for a detector run with and without VerdictAD.

The maps of RUN_DIR (SubspaceAD or detectors.py output on the data/loco layout of tools/prepare_loco.py) are resized bilinearly to the original
image size and written as TIFF to <object>/test/<defect>/<id>.tiff; MVTec's evaluation code (evaluate_experiment.py from the MVTec LOCO AD
download page) then computes the AU-sPRO per object.  With DECISION_DIR the same is done for M' = M * W (merged windows, gamma 0.25).
--anomalydino MAPS_DIR scores AnomalyDINO's own maps (<cls>/test/<good|bad>/<file>.npy, resized to 672 px and smoothed with sigma 4) instead of RUN_DIR's.

usage: python tools/loco_spro.py RUN_DIR LOCO_ROOT MVTEC_EVAL_DIR OUT_DIR [--decisions DECISION_DIR] [--anomalydino MAPS_DIR] [--gamma 0.25] [--workers 8]"""
import argparse, glob, json, os, pickle, shutil, signal, subprocess, sys, time
import numpy as np, cv2, tifffile
from scipy.ndimage import gaussian_filter
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from verdictad import apply_windows

p = argparse.ArgumentParser()
p.add_argument("run_dir"); p.add_argument("loco_root"); p.add_argument("eval_dir"); p.add_argument("out")
p.add_argument("--decisions", default=""); p.add_argument("--anomalydino", default=""); p.add_argument("--gamma", type=float, default=0.25); p.add_argument("--workers", type=int, default=8)
a = p.parse_args()
OBJECTS = ["breakfast_box", "juice_bottle", "pushpins", "screw_bag", "splicing_connectors"]
key = lambda q: "/".join(str(q).replace("\\", "/").split("/")[-4:])


def official(path):
    """data/loco/<obj>/test/{good,bad}/<obj>_<type>_<id>.png -> (<obj>, <type>, <id>) of the official release."""
    q = path.replace("\\", "/").split("/"); obj = q[-4]; typ, idx = os.path.splitext(q[-1])[0][len(obj) + 1:].rsplit("_", 1)
    return obj, typ, idx


def evaluate(maps_dir, obj, out_dir):
    """Runs MVTec's evaluate_experiment.py; it can hang in its worker pool after writing metrics.json, so it is stopped once the file is complete."""
    pr = subprocess.Popen([sys.executable, "evaluate_experiment.py", "--object_name", obj, "--dataset_base_dir", a.loco_root, "--anomaly_maps_dir", maps_dir,
                           "--output_dir", out_dir, "--num_parallel_workers", str(a.workers)], cwd=a.eval_dir, stdout=subprocess.DEVNULL, start_new_session=True)
    f = f"{out_dir}/metrics.json"
    while pr.poll() is None:
        time.sleep(10)
        if os.path.exists(f):
            try:
                json.load(open(f)); time.sleep(5); os.killpg(pr.pid, signal.SIGTERM); break
            except ValueError:
                pass
    pr.wait()
    return json.load(open(f))["localization"]["auc_spro"]


variants = ["detector"] + (["verdictad"] if a.decisions else [])
res = {v: {} for v in variants}
for obj in OBJECTS:
    z = np.load(f"{a.run_dir}/imgmaps_{obj}.npz"); paths = [str(q) for q in z["paths"]]; S = z["maps"].shape[-1]
    if a.anomalydino:
        M = np.stack([gaussian_filter(cv2.resize(np.load(f"{a.anomalydino}/{'/'.join(key(q).split('/')[:3])}/{os.path.splitext(os.path.basename(q))[0]}.npy").astype(np.float32), (S, S)), 4) for q in paths])
    else:
        M = z["maps"].astype(np.float32)
    maps = {"detector": M}
    if a.decisions:
        f = f"{a.decisions}/{obj}.json"; rows = json.load(open(f)) if os.path.exists(f) else pickle.load(open(f"{a.decisions}/{obj}.pkl", "rb"))
        R = {key(r["img_path"]): r for r in rows}
        maps["verdictad"] = np.stack([apply_windows(m, R[key(q)]["boxes"], np.asarray(R[key(q)]["logits"]), a.gamma) for m, q in zip(M, paths)])
    for v in variants:
        tdir = f"{a.out}/tiff_{v}"
        for m, q in zip(maps[v], paths):
            o, typ, idx = official(q); h, w = cv2.imread(f"{a.loco_root}/{o}/test/{typ}/{idx}.png", cv2.IMREAD_UNCHANGED).shape[:2]
            os.makedirs(f"{tdir}/{o}/test/{typ}", exist_ok=True)
            tifffile.imwrite(f"{tdir}/{o}/test/{typ}/{idx}.tiff", cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32))
        res[v][obj] = evaluate(tdir, obj, f"{a.out}/metrics_{v}/{obj}")
        shutil.rmtree(f"{tdir}/{obj}")
        print(v, obj, "AU-sPRO@0.05 logical %.1f structural %.1f mean %.1f" % tuple(100 * res[v][obj][k]["0.05"] for k in ("logical_anomalies", "structural_anomalies", "mean")), flush=True)
json.dump(res, open(f"{a.out}/auc_spro.json", "w"), indent=1)
for lim in ("0.05", "0.3"):
    for v in variants:
        print(f"MEAN {v:10s} AU-sPRO@{lim}  logical / structural / mean", " / ".join(f"{100 * np.mean([res[v][o][k][lim] for o in OBJECTS]):.1f}" for k in ("logical_anomalies", "structural_anomalies", "mean")))
