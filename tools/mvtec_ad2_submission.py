"""Submission folders for the MVTec AD 2 benchmark server (test_private and test_private_mixed), for the detector alone and with VerdictAD.

Each split is a SubspaceAD run on the layout of tools/prepare_mvtec_ad2.py with its decide.py output.  Per class, the threshold of each method is the
mean + 3 std of its pixel scores on the validation split (anomaly-free).  Maps are resized bilinearly to the original image size and written as
anomaly_images/<cls>/<split>/<name>.tiff (float16) and anomaly_images_thresholded/<cls>/<split>/<name>.png (0 / 255).
Check and compress the folders with MVTec's submission script before uploading.

usage: python tools/mvtec_ad2_submission.py OUT_DIR --val RUN DECISIONS --private RUN DECISIONS --mixed RUN DECISIONS [--gamma 0.25]"""
import argparse, json, os, pickle, sys
import numpy as np, cv2, tifffile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from verdictad import apply_windows

p = argparse.ArgumentParser(); p.add_argument("out")
for s in ("val", "private", "mixed"): p.add_argument(f"--{s}", nargs=2, required=True, metavar=("RUN_DIR", "DECISION_DIR"))
p.add_argument("--gamma", type=float, default=0.25); a = p.parse_args()
CLASSES = ["can", "fabric", "fruit_jelly", "rice", "sheet_metal", "vial", "wallplugs", "walnuts"]
key = lambda q: "/".join(str(q).replace("\\", "/").split("/")[-4:])


def load(run, dec, cls):
    z = np.load(f"{run}/imgmaps_{cls}.npz"); paths = [str(q) for q in z["paths"]]; M = z["maps"].astype(np.float32)
    f = f"{dec}/{cls}.json"; rows = json.load(open(f)) if os.path.exists(f) else pickle.load(open(f"{dec}/{cls}.pkl", "rb")); R = {key(r["img_path"]): r for r in rows}
    return {"detector": M, "verdictad": np.stack([apply_windows(m, R[key(q)]["boxes"], np.asarray(R[key(q)]["logits"]), a.gamma) for m, q in zip(M, paths)])}, paths


thresholds = {}
for cls in CLASSES:
    val, _ = load(*a.val, cls); thr = {m: float(v.mean() + 3 * v.std()) for m, v in val.items()}; thresholds[cls] = thr
    for split, (run, dec) in (("test_private", a.private), ("test_private_mixed", a.mixed)):
        maps, paths = load(run, dec, cls)
        for i, q in enumerate(paths):
            h, w = cv2.imread(q, cv2.IMREAD_UNCHANGED).shape[:2]; name = os.path.splitext(os.path.basename(q))[0]
            for m in maps:
                x = cv2.resize(maps[m][i], (w, h), interpolation=cv2.INTER_LINEAR)
                for sub, arr, ext in (("anomaly_images", x.astype(np.float16), "tiff"), ("anomaly_images_thresholded", ((x >= thr[m]) * 255).astype(np.uint8), "png")):
                    d = f"{a.out}/{m}/{sub}/{cls}/{split}"; os.makedirs(d, exist_ok=True)
                    tifffile.imwrite(f"{d}/{name}.tiff", arr) if ext == "tiff" else cv2.imwrite(f"{d}/{name}.png", arr)
    print(cls, {m: round(t, 4) for m, t in thr.items()}, flush=True)
json.dump(thresholds, open(f"{a.out}/thresholds.json", "w"), indent=1)
