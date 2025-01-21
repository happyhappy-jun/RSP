DATASET=$1
METHOD=$2
MODEL=$3
EPOCHS=$4
TOPK=$5
NEIGHBOR=$6
QUEUE=$7


python main_video_segmentation_vip.py \
--finetune ./logs/${DATASET}/${METHOD}/checkpoint-${EPOCHS}.pth \
--output_dir /mnt/ssd0/logs/${DATASET}/${METHOD}/vip_560_560_seg_${EPOCHS}epoch_${TOPK}_${NEIGHBOR}_${QUEUE} \
--data_path /data/VIP_560_560 \
--topk ${TOPK} --size_mask_neighborhood ${NEIGHBOR} --n_last_frames ${QUEUE} \
--model vit_${MODEL}

python ./ATEN/evaluate/test_inst_part_ap.py \


python ./davis2017-evaluation/evaluation_method.py \
--task semi-supervised \
--results_path /mnt/ssd0/logs/${DATASET}/${METHOD}/vip_560_560_seg_${EPOCHS}epoch_${TOPK}_${NEIGHBOR}_${QUEUE} \
--davis_path /data/VIP_560_560 \
--dataset vip