"""Run the Jev decision model on every test image of one detector run.

For each test image it stores the reference-free logit z_free, the reference-conditioned logit z_ref, and 32 window logits
(16 crop windows + 16 red-box windows, side 0.4, stride 0.2, 448 px), all with the same k normal references the detector used.

usage: python decide.py RUN_DIR OUT_DIR --data_root DATA_ROOT [--classes a,b] [--batch 12]
  RUN_DIR holds imgmaps_<cls>.npz (test paths, masks) and refs_<cls>.json (reference paths), as written by patched SubspaceAD or detectors.py.
  OUT_DIR/<cls>.pkl: list of dict(img_path, cls_name, anomaly, z_free, z_ref, boxes, logits)."""
import argparse, glob, json, os, pickle, time
import numpy as np
from PIL import ImageDraw
from verdictad.decider import Decider, Q_CROP, Q_BOX, Q_FREE, Q_REF, _open, _mosaic, _crop, _resize, grid_boxes

p = argparse.ArgumentParser()
p.add_argument("run_dir"); p.add_argument("out"); p.add_argument("--data_root", required=True)
p.add_argument("--classes", default=""); p.add_argument("--batch", type=int, default=12); p.add_argument("--shots", type=int, default=4)
a = p.parse_args(); os.makedirs(a.out, exist_ok=True)

D = Decider(device="cuda", batch=a.batch)
TASK_FREE, TASK = "Inspect this industrial product photo.", "Inspect an industrial product against normal references."
P = 448; boxes = grid_boxes(0.4, 0.2)
rel = lambda q: "/".join(q.replace("\\", "/").split("/")[-4:])          # <cls>/<split>/<type>/<file>
fix = lambda q: q if os.path.exists(q) else os.path.join(a.data_root, rel(q))

cls_list = [c for c in a.classes.split(",") if c] or sorted(os.path.basename(f)[len("imgmaps_"):-4] for f in glob.glob(a.run_dir + "/imgmaps_*.npz"))
for cls in cls_list:
    out = f"{a.out}/{cls}.pkl"
    if os.path.exists(out): continue
    z = np.load(f"{a.run_dir}/imgmaps_{cls}.npz"); paths = [str(x) for x in z["paths"]]; anom = [int(g.any()) for g in z["gts"]]
    refs = [_open(fix(r)) for r in json.load(open(f"{a.run_dir}/refs_{cls}.json"))[:a.shots]]; full = _mosaic(refs, P)
    rows, t0 = [], time.time()
    for i, path in enumerate(paths):
        img = _open(fix(path))
        recs = [(TASK_FREE, [_resize(img, 896)], Q_FREE), (TASK, [_resize(img, 896), _mosaic(refs, 896)], Q_REF)]
        for b in boxes:
            recs.append((TASK, [_resize(_crop(img, b), P), _mosaic([_crop(r, b) for r in refs], P)], Q_CROP))
        for b in boxes:
            w = _resize(img, P).copy()
            ImageDraw.Draw(w).rectangle([int(b[0] * P), int(b[1] * P), int(b[2] * P) - 1, int(b[3] * P) - 1], outline=(255, 0, 0), width=4)
            recs.append((TASK, [w, full], Q_BOX))
        lg = D._logits(recs)
        rows.append(dict(img_path=rel(path), cls_name=cls, anomaly=anom[i], z_free=lg[0], z_ref=lg[1], boxes=boxes * 2, logits=lg[2:]))
        if i % 20 == 0: print(cls, i, len(paths), round(time.time() - t0, 1), flush=True)
    pickle.dump(rows, open(out, "wb")); print(cls, "done", len(paths), round(time.time() - t0, 1), flush=True)
