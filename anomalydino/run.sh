#!/bin/bash
# AnomalyDINO + VerdictAD, scored with AnomalyDINO's own metric code.
# usage: anomalydino/run.sh ANOMALYDINO_REPO DATASET(MVTec|VisA|LOCO) DATA_ROOT K SEED REFS_DIR DECISION_DIR
#   REFS_DIR     = refs_<cls>.json of the SubspaceAD run (same references for every detector)
#   DECISION_DIR = decide.py output for that SubspaceAD run
set -e
AD=$1; DS=$2; DATA=$3; K=$4; S=$5; REFS=$(realpath $6); DEC=$(realpath $7); HERE=$(dirname $(realpath $0))
TMP=$(mktemp -d); python $HERE/convert_units.py $DEC $(echo $DS | tr A-Z a-z) $TMP/u
(cd $AD && AD_REFS_DIR=$REFS python run_anomalydino.py --dataset $DS --data_root $DATA --shots $K --just_seed $S --num_seeds 1 \
   --preprocess agnostic --resolution 672 --eval_segm --no-save_examples --faiss_on_cpu --tag verdictad)
MAPS=$AD/results_$DS/dinov2_vits14_672/${K}-shot_preprocess=agnostic_verdictad/anomaly_maps/seed=$S
KIND=$DS; [ $DS = LOCO ] && KIND=MVTec
python $HERE/ad_jev_image.py --repo $AD --dataset $DS --data_root $DATA --maps_dir $MAPS --logit_u $TMP/u_free --logit_r $TMP/u_ref --exps 0,0.5
CL=$(ls $DEC | sed 's/\.pkl$//' | paste -sd,)
for G in 0 0.25; do
  python $HERE/ad_jev_pixel.py --repo $AD --data_root $DATA --maps_dir $MAPS --region_dir $TMP/u_win --classes $CL --gamma $G --kind $KIND
done
