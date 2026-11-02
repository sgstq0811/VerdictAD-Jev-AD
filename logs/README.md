# Logs

One file per detector, dataset, shot and seed: `<detector>/<dataset>_k<shots>_s<seed>.log`. Seeds 42/43/44 for 1- and 2-shot, 42/43/45 for 4-shot.

- `subspacead/`, `dinov2l/`, `patchcore/`: output of `evaluate.py` with α = 0.5, γ = 0.25 and both window types. Each line holds a class; the last two lines are the class means of the detector alone and with VerdictAD (labelled `SubspaceAD` and `+ decision layer` by the version that produced them).
- `anomalydino/`: output of `anomalydino/run.sh` (AnomalyDINO's own metric code); `exponent 0` and `gamma 0` are AnomalyDINO alone.
- `ablation/`: all ablations on the 4-shot, seed-42 SubspaceAD run, per class and class mean (`img|...`: Image AUROC, Image AUPR; `pix|<windows>|<gamma>`: Pixel AUROC, AUPRO).

`python tools/summarize_logs.py` prints the main tables from these files.
