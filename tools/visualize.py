"""Qualitative panels in the style of PromptAD's plot_sample_cv2: images and maps resized to 400 x 400 (cubic), maps smoothed with a Gaussian of
sigma 4, JET colour map blended 0.5 / 0.5 with the image, ground truth painted red on the image.

For each requested test image of a detector run, writes <out>/<cls>_<defect>_<name>/:
  ref.jpg (2 x 2 mosaic of the references), input.jpg, gt.jpg,
  detector_<norm>.jpg and verdictad_<norm>.jpg for norm in
    class  (map - class min) / class max of that method, as in PromptAD
    image  per-image min-max
  detector_seg.jpg, verdictad_seg.jpg   pixels above half of the per-image min-max range painted red

usage: python tools/visualize.py RUN_DIR DECISION_DIR OUT_DIR CLS FILE [FILE ...]   (FILE: <defect>/<image file name>, e.g. bad/018.JPG)"""
import argparse, json, os, pickle, sys
import numpy as np, cv2
from scipy.ndimage import gaussian_filter
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from verdictad import apply_windows

p = argparse.ArgumentParser(); p.add_argument("run_dir"); p.add_argument("decision_dir"); p.add_argument("out"); p.add_argument("cls"); p.add_argument("files", nargs="+")
p.add_argument("--gamma", type=float, default=0.25); a = p.parse_args()
R4 = 400
key = lambda q: "/".join(str(q).replace("\\", "/").split("/")[-4:])
z = np.load(f"{a.run_dir}/imgmaps_{a.cls}.npz"); paths = [str(q) for q in z["paths"]]; M = z["maps"].astype(np.float32); G = z["gts"]
f = f"{a.decision_dir}/{a.cls}.json"; rows = json.load(open(f)) if os.path.exists(f) else pickle.load(open(f"{a.decision_dir}/{a.cls}.pkl", "rb")); R = {key(r["img_path"]): r for r in rows}
M1 = np.stack([apply_windows(m, R[key(q)]["boxes"], np.asarray(R[key(q)]["logits"]), a.gamma) for m, q in zip(M, paths)])
smooth = lambda v: np.stack([gaussian_filter(cv2.resize(t, (R4, R4), interpolation=cv2.INTER_CUBIC), 4) for t in v])
S = {"detector": smooth(M), "verdictad": smooth(M1)}
refs = [cv2.resize(cv2.imread(q), (R4 // 2, R4 // 2)) for q in json.load(open(f"{a.run_dir}/refs_{a.cls}.json"))[:4]]
mosaic = np.vstack([np.hstack(refs[:2]), np.hstack(refs[2:4])])
blend = lambda u, img: cv2.addWeighted(cv2.applyColorMap(u, cv2.COLORMAP_JET), 0.5, img, 0.5, 0)
for fn in a.files:
    i = [j for j, q in enumerate(paths) if key(q).endswith(fn)][0]
    d = f"{a.out}/{a.cls}_{fn.replace('/', '_').rsplit('.', 1)[0]}"; os.makedirs(d, exist_ok=True)
    img = cv2.resize(cv2.imread(paths[i]), (R4, R4), interpolation=cv2.INTER_CUBIC)
    gt = img.copy(); gt[cv2.resize(G[i].astype(np.uint8), (R4, R4), interpolation=cv2.INTER_NEAREST) > 0] = (0, 0, 255)
    cv2.imwrite(f"{d}/ref.jpg", mosaic); cv2.imwrite(f"{d}/input.jpg", img); cv2.imwrite(f"{d}/gt.jpg", gt)
    for m, v in S.items():
        x = v[i]; u_img = ((x - x.min()) / max(float(x.max() - x.min()), 1e-8) * 255).astype(np.uint8)
        cv2.imwrite(f"{d}/{m}_class.jpg", blend(((x - v.min()) / v.max() * 255).astype(np.uint8), img))
        cv2.imwrite(f"{d}/{m}_image.jpg", blend(u_img, img))
        seg = img.copy(); seg[u_img >= 128] = (0, 0, 255); cv2.imwrite(f"{d}/{m}_seg.jpg", seg)
    print(d, flush=True)
