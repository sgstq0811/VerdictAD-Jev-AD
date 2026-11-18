# Logs

One file per detector, dataset, shot and seed: `<detector>/<dataset>_k<shots>_s<seed>.log`. Seeds 42/43/44 for 1- and 2-shot, 42/43/45 for 4-shot.

- `subspacead/`, `dinov2l/`, `patchcore/`: output of `evaluate.py` with α = 0.5, γ = 0.25 and both window types. Each line holds a class; the last two lines are the class means of the detector alone and with VerdictAD (labelled `SubspaceAD` and `+ decision layer` by the version that produced them).
- `anomalydino/`: output of `anomalydino/run.sh` (AnomalyDINO's own metric code); `exponent 0` and `gamma 0` are AnomalyDINO alone.
- `ablation/`: all ablations on the 4-shot, seed-42 SubspaceAD run, per class and class mean (`img|...`: Image AUROC, Image AUPR; `pix|<windows>|<gamma>`: Pixel AUROC, AUPRO). `dinov2l_layers.json`: DINOv2-L feature layer 8 to 24 with the same references and decisions (`detectors.py --layer`).

- `loco_spro/`: AU-sPRO of MVTec LOCO AD (`tools/loco_spro.py`, MVTec's evaluation code on the original release), 4-shot, seed 42, per object and FPR limit (0.01 to 1.0) for logical anomalies, structural anomalies and their mean; `<detector>_verdictad` adds VerdictAD.
- `mvtec_ad2/`: MVTec AD 2 public test split, SubspaceAD 4-shot, seed 42, per class: Image AUROC / AUPR, Pixel AUROC, AU-PRO at FPR 0.3 and 0.05, for the detector and with VerdictAD; `decisions_only` is the Image AUROC of the two image decisions alone.

`python tools/summarize_logs.py` prints the main tables from these files.
