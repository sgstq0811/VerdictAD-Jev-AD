"""Jev-family decision model (Cloudflare Clef-flash) as a training-free decision layer.

Three kinds of Noul decisions, each returning one logit (log-odds of "has a defect"):
  image_logit  - the query image alone                                       (reference-free)
  ref_logit    - the query image and a 2x2 mosaic of the k normal references  (reference-conditioned)
  window_logits- a grid of windows (side 0.4, stride 0.2 by default), judged either
                 'crop': query crop against the same-location crops of the references, or
                 'box' : the full query with the window drawn as a red box against the full references, or
                 'merged': both, concatenated.
"""
import sys
import torch
from PIL import Image, ImageDraw
from huggingface_hub import snapshot_download

Q_FREE = "Does the object in the image have a defect, damage, or contamination?"
Q_REF = ("Image 1 shows an object to inspect. Image 2 is a 2x2 grid of four normal reference samples of the same kind of object. "
         "Compared with the normal references, does the object in image 1 have a defect, damage, contamination, or a wrong arrangement?")
Q_CROP = ("Image 1 is a close-up crop of an object to inspect. Image 2 is a 2x2 grid of crops at the same location from four normal reference samples of the same kind of object. "
          "Compared with the normal references, does the crop in image 1 contain a defect, damage, contamination, or a wrong arrangement?")
Q_BOX = ("Image 1 is an object to inspect; a red box marks one area. Image 2 is a 2x2 grid of four normal reference samples of the same kind of object. "
         "Compared with the normal references, does the area inside the red box of image 1 show a defect, damage, contamination, a missing or extra item, or a wrong arrangement?")


def _open(x):
    return x.convert("RGB") if isinstance(x, Image.Image) else Image.open(x).convert("RGB")


def _resize(img, px):
    return img.resize((px, px), Image.BICUBIC)


def _mosaic(imgs, px):
    """2x2 mosaic of px x px; with fewer than four images they repeat cyclically."""
    h = px // 2
    out = Image.new("RGB", (px, px))
    for i in range(4):
        out.paste(_resize(imgs[i % len(imgs)], h), ((i % 2) * h, (i // 2) * h))
    return out


def grid_boxes(side=0.4, stride=0.2):
    n = int(round((1 - side) / stride)) + 1
    return [(ix * stride, iy * stride, ix * stride + side, iy * stride + side) for iy in range(n) for ix in range(n)]


def _crop(img, b):
    W, H = img.size
    return img.crop((int(b[0] * W), int(b[1] * H), int(b[2] * W), int(b[3] * H)))


class Decider:
    def __init__(self, model="Cloudflare/clef-flash", device="cuda", batch=8):
        path = snapshot_download(model)
        sys.path.insert(0, path)
        from joint_schema_model import collate_records, encode_record, load_release_model
        self._collate, self._encode = collate_records, encode_record
        self.model, self.processor = load_release_model(path, device=device)
        self.device, self.batch = torch.device(device), batch

    @torch.inference_mode()
    def _logits(self, records):
        """records: list of (task, images, question); returns one logit per record, `batch` records per forward pass."""
        out = []
        for s in range(0, len(records), self.batch):
            encs = [self._encode(self.processor.tokenizer,
                                 {"state": {"task": task}, "images": images, "questions": {"defect": {"type": "noul", "instructions": q}}},
                                 processor=self.processor) for task, images, q in records[s:s + self.batch]]
            b = self._collate(encs, self.processor.tokenizer.pad_token_id, self.device)
            res = self.model(b)
            for i, enc in enumerate(encs):
                lg = res[i][0].float()
                opt = list(enc.questions[0].option_ids)
                out.append(float(lg[opt.index("true")] - lg[opt.index("false")]))
        return out

    def image_logit(self, img, px=896):
        return self._logits([("Inspect this industrial product photo.", [_resize(_open(img), px)], Q_FREE)])[0]

    def ref_logit(self, img, refs, px=896):
        refs = [_open(r) for r in refs]
        return self._logits([("Inspect an industrial product against normal references.", [_resize(_open(img), px), _mosaic(refs, px)], Q_REF)])[0]

    def window_logits(self, img, refs, mode="merged", side=0.4, stride=0.2, px=448, color=(255, 0, 0)):
        """Returns (boxes, logits).  mode: 'crop' | 'box' | 'merged' (both kinds, boxes repeated)."""
        img, refs = _open(img), [_open(r) for r in refs]
        boxes = grid_boxes(side, stride)
        task = "Inspect an industrial product against normal references."
        recs, kinds = [], []
        if mode in ("crop", "merged"):
            for b in boxes:
                recs.append((task, [_resize(_crop(img, b), px), _mosaic([_crop(r, b) for r in refs], px)], Q_CROP))
        if mode in ("box", "merged"):
            full_refs = _mosaic(refs, px)
            for b in boxes:
                q = _resize(img, px).copy()
                ImageDraw.Draw(q).rectangle([int(b[0] * px), int(b[1] * px), int(b[2] * px) - 1, int(b[3] * px) - 1], outline=color, width=4)
                recs.append((task, [q, full_refs], Q_BOX))
        logits = self._logits(recs)
        return (boxes * (2 if mode == "merged" else 1)), logits
