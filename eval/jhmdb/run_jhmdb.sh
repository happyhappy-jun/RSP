DATASET=$1
METHOD=$2
EPOCHS=$3
TOPK=$4
NEIGHBOR=$5
QUEUE=$6

python jhmdb_evaluation.py \
--resume ./logs/${DATASET}/${METHOD}/checkpoint-${EPOCHS}.pth \
--save-path /home/huiwon/logs/${DATASET}/${METHOD}/jhmdb_seg_${EPOCHS}epoch_${TOPK}_${NEIGHBOR}_${QUEUE} \
--root /data \
--filelist /data/JHMDB/val_list.txt \
--topk ${TOPK} --radius ${NEIGHBOR} --videoLen ${QUEUE}

python jhmdb_evaluation/eval_pck.py \
--src-folder /home/huiwon/logs/${DATASET}/${METHOD}/jhmdb_seg_${EPOCHS}epoch_${TOPK}_${NEIGHBOR}_${QUEUE} \
--root /data \
--filelist /data/JHMDB/val_list.txt