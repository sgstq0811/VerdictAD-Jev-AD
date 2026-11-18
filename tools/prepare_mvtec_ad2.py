"""Lay out one split of the MVTec AD 2 release in the MVTec AD format read by SubspaceAD and decide.py (symbolic links, nothing is copied).

<cls>/train/good                 -> official train/good
<cls>/test/good, test/bad        -> official test_public/{good,bad}             (SPLIT = test_public)
<cls>/ground_truth/bad           -> official test_public/ground_truth/bad
<cls>/test/<SPLIT>               -> official validation/good, test_private or test_private_mixed (no ground truth)

usage: python tools/prepare_mvtec_ad2.py AD2_ROOT OUT_ROOT [--split test_public|validation|test_private|test_private_mixed]"""
import argparse, os

p = argparse.ArgumentParser(); p.add_argument("src"); p.add_argument("dst"); p.add_argument("--split", default="test_public", choices=["test_public", "validation", "test_private", "test_private_mixed"])
a = p.parse_args(); src = os.path.abspath(a.src)
CLASSES = ["can", "fabric", "fruit_jelly", "rice", "sheet_metal", "vial", "wallplugs", "walnuts"]
for c in CLASSES:
    links = {"train/good": f"{src}/{c}/train/good"}
    if a.split == "test_public":
        links.update({"test/good": f"{src}/{c}/test_public/good", "test/bad": f"{src}/{c}/test_public/bad", "ground_truth/bad": f"{src}/{c}/test_public/ground_truth/bad"})
    else:
        links[f"test/{a.split}"] = f"{src}/{c}/validation/good" if a.split == "validation" else f"{src}/{c}/{a.split}"
    for d, s in links.items():
        os.makedirs(os.path.dirname(f"{a.dst}/{c}/{d}"), exist_ok=True)
        if not os.path.lexists(f"{a.dst}/{c}/{d}"):
            os.symlink(s, f"{a.dst}/{c}/{d}")
    print(c, {d: len(os.listdir(f"{a.dst}/{c}/{d}")) for d in links}, flush=True)
