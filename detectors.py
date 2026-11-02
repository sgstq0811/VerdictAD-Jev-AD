"""Reference-memory detectors (DINOv2-L layer-18 nearest neighbour, PatchCore WRN50, SPADE WRN50) with the references of a SubspaceAD run.
Writes imgmaps_<cls>.npz (maps at 672 px, gaussian sigma 4; gts; paths), imgscores_<cls>.json (image score = max of the patch distance map)
and refs_<cls>.json in the SubspaceAD run format, so decide.py and evaluate.py run on them unchanged.
usage: python detectors.py dino|pc|spade REF_DIR TEST_DIR OUT_DIR --data_root DATA_ROOT [--shots K] [--layer 18]
  TEST_DIR = a SubspaceAD run directory (test paths and masks);  REF_DIR = folder with refs_<cls>.json."""
import argparse, glob, json, os, time
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter
p = argparse.ArgumentParser(); p.add_argument("head"); p.add_argument("ref_dir"); p.add_argument("test_dir"); p.add_argument("out")
p.add_argument("--shots", type=int, default=4); p.add_argument("--classes", default=""); p.add_argument("--data_root", default=""); p.add_argument("--layer", type=int, default=18); p.add_argument("--spade_img", default="max", choices=["max", "knn"])
a = p.parse_args(); os.makedirs(a.out, exist_ok=True); S = 672
dev = "cuda"
if a.head == "dino":
    from transformers import AutoModel
    net = AutoModel.from_pretrained("facebook/dinov2-large", dtype=torch.float16).to(dev).eval(); PX = 672; LAYER = a.layer
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1); std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    @torch.inference_mode()
    def feats(img):
        x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.).permute(2, 0, 1)[None].to(dev)
        h = net(pixel_values=((x - mean) / std).half(), output_hidden_states=True).hidden_states[LAYER]
        return F.normalize(h[0, 1:].float(), dim=-1), (PX // 14, PX // 14)
    def dist(f, bank): return 1 - (f @ bank.T).max(dim=1).values
elif a.head == "spade":   # SPADE (Cohen & Hoshen 2020) as in SPADE-pytorch: WRN50 (torchvision weights), 224 px, top-k references by global layer-4 feature, pixel NN over layer1-3 of those references
    import timm
    net = timm.create_model("wide_resnet50_2.tv_in1k", pretrained=True, features_only=True, out_indices=(1, 2, 3, 4)).to(dev).eval(); PX = 224; TOPK = 5
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1); std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    @torch.inference_mode()
    def feats(img):
        x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.).permute(2, 0, 1)[None].to(dev); f1, f2, f3, f4 = net((x - mean) / std)
        f = torch.cat([f1] + [F.interpolate(t, size=f1.shape[-2:], mode="nearest") for t in (f2, f3)], 1)[0]; h, w = f.shape[-2:]
        return (f.permute(1, 2, 0).reshape(h * w, -1), f4.mean((2, 3))[0]), (h, w)
else:
    import timm
    net = timm.create_model("wide_resnet50_2.racm_in1k", pretrained=True, features_only=True, out_indices=(2, 3)).to(dev).eval(); PX = 512
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1); std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    @torch.inference_mode()
    def feats(img):
        x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.).permute(2, 0, 1)[None].to(dev); f2, f3 = net((x - mean) / std)
        f2 = F.avg_pool2d(f2, 3, 1, 1); f3 = F.avg_pool2d(f3, 3, 1, 1); f3 = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        f = torch.cat([f2, f3], 1)[0]; h, w = f.shape[-2:]; return f.permute(1, 2, 0).reshape(h * w, -1), (h, w)
    def dist(f, bank): return torch.cdist(f, bank).min(dim=1).values
def load(p_): return Image.open(p_).convert("RGB").resize((PX, PX), Image.BICUBIC)
fix = lambda q: q if os.path.exists(q) else os.path.join(a.data_root, *q.replace("\\", "/").split("/")[-4:]) if a.data_root else q
cls_list = [c for c in a.classes.split(",") if c] or sorted(os.path.basename(f)[len("imgmaps_"):-4] for f in glob.glob(a.test_dir + "/imgmaps_*.npz"))
for cls in cls_list:
    if os.path.exists(f"{a.out}/imgscores_{cls}.json"): continue
    z = np.load(f"{a.test_dir}/imgmaps_{cls}.npz"); paths = [str(x) for x in z["paths"]]; gts = z["gts"]
    refs = json.load(open(f"{a.ref_dir}/refs_{cls}.json"))[:a.shots]; json.dump(refs, open(f"{a.out}/refs_{cls}.json", "w"))
    if a.head == "spade":
        rf = [feats(load(fix(r)))[0] for r in refs]; gal = torch.stack([g for _, g in rf]); pix = torch.stack([p_ for p_, _ in rf])
    else: bank = torch.cat([feats(load(fix(r)))[0] for r in refs], 0)
    maps = []; sc = []; t0 = time.time(); n = len(paths)
    for i in range(n):
        f, (h, w) = feats(load(fix(paths[i])))
        if a.head == "spade":
            gd = torch.cdist(f[1][None], gal)[0]; k = min(TOPK, len(refs)); dk, ik = gd.topk(k, largest=False)
            d = torch.cdist(f[0], pix[ik].reshape(-1, pix.shape[-1])).min(dim=1).values.reshape(1, 1, h, w).float(); sc.append(float(dk.mean()) if a.spade_img == "knn" else float(d.max()))
        else:
            d = dist(f, bank).reshape(1, 1, h, w).float(); sc.append(float(d.max()))
        m = F.interpolate(d, size=(S, S), mode="bilinear", align_corners=False)[0, 0].cpu().numpy(); maps.append(gaussian_filter(m, sigma=4).astype(np.float16))
    np.savez_compressed(f"{a.out}/imgmaps_{cls}.npz", maps=np.stack(maps), gts=gts[:n], paths=np.array(paths[:n]))
    json.dump(dict(paths=paths[:n], img_true=[int(g.any()) for g in gts[:n]], img_pred=sc), open(f"{a.out}/imgscores_{cls}.json", "w")); print(cls, n, round(time.time() - t0, 1), flush=True)
