<div align="center">

# VerdictAD: Few-Shot Anomaly Detection via Decision-Guided Refinement

[![arXiv](https://img.shields.io/badge/arXiv-2611.02654-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2611.02654)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-VerdictAD-ffd21e)](https://huggingface.co/datasets/sunsean/VerdictAD)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](requirements.txt)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11-EE4C2C?logo=pytorch&logoColor=white)](requirements.txt)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue)](LICENSE)

</div>

<p align="center"><img src="assets/pipeline.png" width="95%"></p>

<details>
<summary><b>Abstract</b> (click to expand)</summary>

Detecting defects in real-world industrial inspection often requires learning from only a few normal images per category. Recent few-shot anomaly detection methods have achieved strong performance by measuring feature differences between query images and normal references. However, visual differences do not always indicate genuine defects, as normal appearance variations can also produce high anomaly scores. Conversely, structural or logical anomalies may not produce strong local feature differences, making them difficult to detect. To address this limitation, we introduce VerdictAD, a training-free framework that refines existing anomaly detectors through visual decisions. VerdictAD adopts a typed decision model to assess the query image and its local regions against a few normal references. By directly producing structured decision probabilities without text generation, the model provides complementary evidence for refining anomaly responses at different spatial scales. Building on this capability, we introduce two modules: Global Verdict Refinement (GVR) uses image-level decisions to refine the overall anomaly score, while Local Verdict Refinement (LVR) aggregates regional decisions to refine the anomaly map. Combined with a simple DINOv2-based detector, VerdictAD achieves competitive performance in one-shot and few-shot anomaly detection. As a plug-and-play framework, it consistently improves existing anomaly detectors across MVTec AD, VisA, and MVTec LOCO AD in one-shot and few-shot settings.

</details>

## 🚀 News

- **2026.11** AU-sPRO on the official MVTec LOCO AD release, MVTec AD 2 results, submission and visualization tools.
- **2026.11** Paper on [arXiv](https://arxiv.org/abs/2611.02654).
- **2026.11** Code, experiment logs and [decision cache](https://huggingface.co/datasets/sunsean/VerdictAD) released.

## 📊 Results

4-shot, mean over three reference draws (seeds 42, 43, 45). Each cell: Image AUROC / Pixel AUROC / AUPRO (%). The detector and VerdictAD use the same references.

| Detector | MVTec AD | VisA | MVTec LOCO AD |
|---|:-:|:-:|:-:|
| PatchCore | 93.0 / 96.4 / 91.2 | 89.0 / 97.5 / 89.4 | 67.4 / 74.9 / 66.7 |
| **+ VerdictAD** | **97.5 / 96.5 / 92.0** | **93.9 / 98.0** / 88.5 | **78.0 / 79.9 / 68.7** |
| DINOv2-L | 97.2 / 97.4 / 95.2 | 93.6 / 98.3 / 93.8 | 68.7 / 77.1 / 62.8 |
| **+ VerdictAD** | **98.7 / 97.9 / 95.7** | **96.9 / 98.7 / 94.4** | **77.5 / 80.0 / 65.4** |
| AnomalyDINO | 98.0 / 97.4 / 93.7 | 93.6 / 98.2 / 94.6 | 67.4 / 81.5 / 59.5 |
| **+ VerdictAD** | **98.9 / 97.7 / 94.2** | **96.3 / 98.3 / 95.2** | **74.9 / 82.7 / 61.1** |
| SubspaceAD | 98.1 / 98.0 / 95.2 | 94.6 / 98.4 / 95.9 | 67.5 / 76.6 / 69.5 |
| **+ VerdictAD** | **98.8 / 98.1 / 95.6** | **96.5 / 98.8 / 96.3** | **75.3 / 79.3 / 71.3** |

The 1-shot and 2-shot tables, Image AUPR, and every per-seed, per-class result are in [`logs/`](logs). `python tools/summarize_logs.py` rebuilds all tables from them.

**MVTec LOCO AD, AU-sPRO.** Official sPRO metric of MVTec LOCO AD, computed with MVTec's evaluation code on the original release (4-shot, seed 42, class mean, FPR limit 0.05 / 0.3). Each cell: logical / structural / mean (%).

| Detector | AU-sPRO@0.05 | AU-sPRO@0.3 |
|---|:-:|:-:|
| PatchCore | 47.6 / 66.9 / 57.3 | 77.3 / 88.7 / 83.0 |
| **+ VerdictAD** | **51.2 / 67.1 / 59.1** | **82.1 / 89.1 / 85.6** |
| DINOv2-L | 42.2 / 69.9 / 56.0 | 77.8 / 87.1 / 82.4 |
| **+ VerdictAD** | **48.7 / 72.0 / 60.3** | **81.3 / 88.7 / 85.0** |
| AnomalyDINO | 45.1 / 64.8 / 54.9 | 77.4 / 85.1 / 81.3 |
| **+ VerdictAD** | **46.1 / 65.8 / 56.0** | **78.4 / 86.1 / 82.2** |
| SubspaceAD | 41.9 / 75.5 / 58.7 | 78.6 / 89.6 / 84.1 |
| **+ VerdictAD** | **44.5 / 75.9 / 60.2** | **80.5 / 90.1 / 85.3** |

**MVTec AD 2, public test split.** SubspaceAD 4-shot, seed 42, all VerdictAD settings unchanged; metrics on the 672 px maps.

| | Image AUROC | Image AUPR | Pixel AUROC | AU-PRO@0.3 | AU-PRO@0.05 |
|---|:-:|:-:|:-:|:-:|:-:|
| SubspaceAD | 68.2 | 78.1 | 90.8 | **64.9** | 40.0 |
| **+ VerdictAD** | **71.7** | **80.7** | **90.9** | 64.7 | **40.2** |

Per-object and per-class values are in [`logs/loco_spro`](logs/loco_spro) and [`logs/mvtec_ad2`](logs/mvtec_ad2).

## ⚙️ Installation

```bash
git clone https://github.com/sgstq0811/VerdictAD-Jev-AD.git && cd VerdictAD-Jev-AD
pip install -r requirements.txt
git clone https://github.com/CLendering/SubspaceAD.git && (cd SubspaceAD && git apply ../patches/subspacead.patch)
```

The patch makes SubspaceAD save its anomaly maps, image scores and sampled references. The decision model [Clef-Flash 9B](https://huggingface.co/Cloudflare/clef-flash) is downloaded from Hugging Face on first use.

## 🧩 Datasets

| Dataset | Source | Layout |
|---|---|---|
| MVTec AD | [mvtec.com](https://www.mvtec.com/company/research/datasets/mvtec-ad) | as released |
| VisA | [spot-diff](https://github.com/amazon-science/spot-diff) | 1-class split (`split_csv/1cls.csv`) |
| MVTec LOCO AD | [mvtec.com](https://www.mvtec.com/company/research/datasets/mvtec-loco) | `python tools/prepare_loco.py <loco_root> data/loco` |
| MVTec AD 2 | [mvtec.com](https://www.mvtec.com/company/research/datasets/mvtec-ad-2) | `python tools/prepare_mvtec_ad2.py <ad2_root> data/ad2 [--split ...]` |

## 🛠️ Usage

**1. Base detector.** SubspaceAD, 4-shot, seed 42 (`--dataset_name visa` for VisA; `mvtec_ad` with `data/loco` for LOCO):

```bash
cd SubspaceAD && SUB_NO_ATTN=1 python main.py --dataset_name mvtec_ad --dataset_path ../data/mvtec_anomaly_detection \
  --k_shot 4 --seed 42 --image_res 672 --layers=-12,-13,-14,-15,-16,-17,-18 --model_ckpt facebook/dinov2-with-registers-giant \
  --aug_count 30 --pca_ev 0.99 --agg_method mean --outdir ../runs/sub_mvtec_k4_s42 && cd ..
RUN=$(ls -d runs/sub_mvtec_k4_s42/*/)
```

PatchCore (`pc`) and DINOv2-L (`dino`) take the same references:

```bash
python detectors.py dino $RUN $RUN runs/dino_mvtec_k4_s42 --shots 4 --data_root data/mvtec_anomaly_detection
```

**2. Decisions.** GVR and LVR decisions for every test image (`--batch 4` fits a 24 GB GPU):

```bash
python decide.py $RUN runs/jev_mvtec_k4_s42 --data_root data/mvtec_anomaly_detection
```

**3. Evaluation.** Detector alone and with VerdictAD (α = 0.5, γ = 0.25; AUPRO uses SubspaceAD's official function):

```bash
python evaluate.py $RUN runs/jev_mvtec_k4_s42 --subspacead SubspaceAD
python evaluate.py runs/dino_mvtec_k4_s42 runs/jev_mvtec_k4_s42 --subspacead SubspaceAD
```

The decisions depend only on the query and its references, so one `decide.py` output serves every detector that uses those references. The decisions for all 27 runs in the paper are on [Hugging Face](https://huggingface.co/datasets/sunsean/VerdictAD); `evaluate.py` reads them directly, so step 2 can be skipped. `--alpha`, `--gamma` and `--windows crop|box|merged` reproduce the ablations.

**AnomalyDINO** is scored with its own metric code. Apply `patches/anomalydino.patch` to [AnomalyDINO](https://github.com/dammsi/AnomalyDINO), then run:

```bash
bash anomalydino/run.sh /path/to/AnomalyDINO MVTec data/mvtec_anomaly_detection 4 42 $RUN runs/jev_mvtec_k4_s42
```

**AU-sPRO on MVTec LOCO AD.** Download MVTec's evaluation code from the [MVTec LOCO AD page](https://www.mvtec.com/company/research/datasets/mvtec-loco) and pass the original release (not `data/loco`):

```bash
python tools/loco_spro.py $RUN /path/to/mvtec_loco_anomaly_detection /path/to/mvtec_loco_ad_evaluation runs/spro_loco --decisions runs/jev_loco_k4_s42
```

The script writes the maps of the detector and of VerdictAD as TIFF at the original resolution, runs MVTec's `evaluate_experiment.py` per object and deletes the TIFFs afterwards. `--anomalydino <maps_dir>` scores AnomalyDINO's saved maps instead of `$RUN`'s.

**MVTec AD 2.** Lay out each split, run SubspaceAD and `decide.py` on it as above, and score the public test split; `--fpr_limit 0.05` gives the benchmark's AU-PRO:

```bash
python tools/prepare_mvtec_ad2.py /path/to/mvtec_ad_2 data/ad2                      # test_public; repeat with --split validation / test_private / test_private_mixed
python evaluate.py $RUN runs/jev_ad2_k4_s42 --subspacead SubspaceAD --fpr_limit 0.05
python tools/mvtec_ad2_submission.py runs/ad2_submission --val $RUN_VAL $DEC_VAL --private $RUN_PRIV $DEC_PRIV --mixed $RUN_MIX $DEC_MIX
```

The submission script thresholds every class at the mean + 3 std of the validation scores and writes the `anomaly_images` and `anomaly_images_thresholded` folders for the detector and for VerdictAD.

**Visualization.** Qualitative panels in the style of PromptAD (400 px, Gaussian σ = 4, JET blended 0.5 / 0.5, ground truth in red), with class-wise and per-image normalization and a segmentation at half of each map's range:

```bash
python tools/visualize.py $RUN runs/jev_loco_k4_s42 runs/vis pushpins bad/pushpins_structural_anomalies_063.png
```

## 📣 Citation

```bibtex
@article{sun2026verdictad,
  title   = {VerdictAD: Few-Shot Anomaly Detection via Decision-Guided Refinement},
  author  = {Sun, Shangquan and Luo, Wenjun and Wang, Yuanzhi and Li, Ruiqi and Chen, Ruoyu and Liu, Jiachen and Lu, Shijian},
  journal = {arXiv preprint arXiv:2611.02654},
  year    = {2026}
}
```

## Acknowledgement

This code builds on [SubspaceAD](https://github.com/CLendering/SubspaceAD), [AnomalyDINO](https://github.com/dammsi/AnomalyDINO), [DINOv2](https://github.com/facebookresearch/dinov2) and [Clef-Flash](https://huggingface.co/Cloudflare/clef-flash).

## 📬 Contact

Please open an issue, or email Shangquan Sun ([shangquansun@gmail.com](mailto:shangquansun@gmail.com)).

## License

Code is released under [Apache 2.0](LICENSE). Datasets and model weights keep their own licenses.
