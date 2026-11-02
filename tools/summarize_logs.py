"""Rebuild the main table (mean ± std over three seeds) from logs/.  usage: python tools/summarize_logs.py [LOG_DIR]"""
import glob, os, re, sys
import numpy as np

root = sys.argv[1] if len(sys.argv) > 1 else "logs"
R = re.compile(r"(?:SubspaceAD|detector)\s+([\d.]+) / ([\d.]+) / ([\d.]+) / ([\d.]+)\n\s+\+ (?:decision layer|VerdictAD)\s+([\d.]+) / ([\d.]+) / ([\d.]+) / ([\d.]+)")
res = {}
for det, name in (("patchcore", "PatchCore"), ("dinov2l", "DINOv2-L"), ("subspacead", "SubspaceAD")):
    for f in glob.glob(f"{root}/{det}/*.log"):
        ds, k, s = re.match(r"(\w+)_k(\d)_s(\d+)", os.path.basename(f)).groups(); v = [float(x) for x in R.search(open(f).read()).groups()]
        res.setdefault((name, ds, k), []).append((v[:4], v[4:]))
for f in glob.glob(f"{root}/anomalydino/*.log"):   # AnomalyDINO's own scorer: image lines per exponent, pixel lines per class and gamma
    ds, k, s = re.match(r"(\w+)_k(\d)_s(\d+)", os.path.basename(f)).groups(); t = open(f).read()
    im = {float(a): (float(b), float(c)) for a, b, c in re.findall(r"exponent ([\d.]+): Image AUROC ([\d.]+)\s+Image AUPR ([\d.]+)", t)}
    px = {}
    for g in ("0", "0.25"):
        seg = t.split(f"== pixel gamma {g}\n")[1].split("==")[0].split("done")[0]
        v = [l.split() for l in seg.split("\n") if re.match(r"^[a-z_0-9]+ [0-9.]+ [0-9.]+$", l)]
        px[g] = (np.mean([float(x[1]) for x in v]), np.mean([float(x[2]) for x in v]))
    res.setdefault(("AnomalyDINO", ds, k), []).append(([*im[0.0], *px["0"]], [*im[0.5], *px["0.25"]]))
fmt = lambda a: " / ".join(f"{m:.2f}±{s:.2f}" for m, s in zip(np.mean(a, 0), np.std(a, 0)))
for k in "124":
    print(f"\n{k}-shot   (Image AUROC / Image AUPR / Pixel AUROC / AUPRO)\n| Method | MVTec AD | VisA | MVTec LOCO AD |\n|---|---|---|---|")
    for m in ("PatchCore", "DINOv2-L", "AnomalyDINO", "SubspaceAD"):
        for i, suf in ((0, ""), (1, " + VerdictAD")):
            print(f"| {m}{suf} | " + " | ".join(fmt(np.array([x[i] for x in res[(m, ds, k)]])) for ds in ("mvtec", "visa", "loco")) + " |")
