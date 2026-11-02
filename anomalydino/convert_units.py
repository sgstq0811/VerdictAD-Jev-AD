"""Converts decision outputs of decide_unit.py / decide_loco_unit.py into the three pickle folders read by ad_jev_image.py and ad_jev_pixel.py (img_path in the AnomalyDINO-era key style).
usage: convert_units.py UNIT_DIR DATASET(mvtec|visa|loco) OUT_PREFIX   -> OUT_PREFIX_free, OUT_PREFIX_ref (logit per image), OUT_PREFIX_win (boxes, logits)"""
import glob, os, pickle, sys
src, ds, pre = sys.argv[1:4]
for d in ("free", "ref", "win"): os.makedirs(f"{pre}_{d}", exist_ok=True)
def conv(p):
    q = p.replace("\\", "/").split("/")   # cls/test/<good|bad|defect>/file
    if ds == "visa": return f"{q[-4]}/Data/Images/{'Normal' if q[-2] == 'good' else 'Anomaly'}/{q[-1]}"
    return p
for f in sorted(glob.glob(src + "/*.pkl")):
    c = os.path.basename(f)[:-4]; rows = pickle.load(open(f, "rb")); mk = lambda **kw: [dict(img_path=conv(r["img_path"]), cls_name=c, anomaly=r["anomaly"], **{k: fn(r) for k, fn in kw.items()}) for r in rows]
    pickle.dump(mk(logit=lambda r: r["z_free"]), open(f"{pre}_free/{c}.pkl", "wb")); pickle.dump(mk(logit=lambda r: r["z_ref"]), open(f"{pre}_ref/{c}.pkl", "wb"))
    pickle.dump(mk(boxes=lambda r: r["boxes"], logits=lambda r: r["logits"]), open(f"{pre}_win/{c}.pkl", "wb"))
print("converted", len(glob.glob(src + "/*.pkl")), "classes")
