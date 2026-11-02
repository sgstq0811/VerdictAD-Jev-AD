"""Convert the official MVTec LOCO AD release into the MVTec AD layout read by SubspaceAD and AnomalyDINO.

<cls>/train/good/<cls>_NNN.png                        official train/good (the validation split is not used)
<cls>/test/good/<cls>_good_NNN.png                    official test/good
<cls>/test/bad/<cls>_<type>_NNN.png                   official test/{logical,structural}_anomalies
<cls>/ground_truth/bad/<cls>_<type>_NNN_mask.png      union of all official masks of that image, binarised to {0, 255}

usage: python tools/prepare_loco.py LOCO_ROOT OUT_ROOT"""
import os, shutil, sys, glob
import numpy as np
from PIL import Image

src, dst = sys.argv[1], sys.argv[2]
CLASSES = ["breakfast_box", "juice_bottle", "pushpins", "screw_bag", "splicing_connectors"]
for c in CLASSES:
    for d in ("train/good", "test/good", "test/bad", "ground_truth/bad"):
        os.makedirs(f"{dst}/{c}/{d}", exist_ok=True)
    for f in sorted(glob.glob(f"{src}/{c}/train/good/*.png")):
        shutil.copy(f, f"{dst}/{c}/train/good/{c}_{os.path.basename(f)}")
    for f in sorted(glob.glob(f"{src}/{c}/test/good/*.png")):
        shutil.copy(f, f"{dst}/{c}/test/good/{c}_good_{os.path.basename(f)}")
    for t in ("logical_anomalies", "structural_anomalies"):
        for f in sorted(glob.glob(f"{src}/{c}/test/{t}/*.png")):
            n = os.path.splitext(os.path.basename(f))[0]
            shutil.copy(f, f"{dst}/{c}/test/bad/{c}_{t}_{n}.png")
            masks = [np.array(Image.open(m)) > 0 for m in sorted(glob.glob(f"{src}/{c}/ground_truth/{t}/{n}/*.png"))]
            Image.fromarray((np.any(masks, axis=0) * 255).astype(np.uint8)).save(f"{dst}/{c}/ground_truth/bad/{c}_{t}_{n}_mask.png")
    print(c, {d: len(os.listdir(f"{dst}/{c}/{d}")) for d in ("train/good", "test/good", "test/bad")}, flush=True)
