python captioning/extract_frames.py \
  --data_root /home/byungjun/kinetics400/train2 \
  --output_root artifacts \
  --workers 8


  python -m torch.distributed.launch --nproc_per_node=8 main_pretrain_rsp_llm_global.py \
    --batch_size 8 \
    --accum_iter 24 \
    --model rsp_vit_small_patch16 \
    --epochs 400 \
    --warmup_epochs 40 \
    --video_root /home/byungjun/kinetics400/train2 \
    --data_path /home/byungjun/RSP/artifacts/results/frame_analysis_results_complete.json \
    --log_dir logs/rsp-llm-global \
    --output_dir outputs/rsp-llm-global \
    --norm_pix_loss \
    --repeated_sampling 2


  python precompute_embeddings.py \
    --json_path /home/byungjun/RSP/artifacts/results/frame_analysis_results_complete.json \
    --output_dir artifacts/ \
    --batch_size 32


  python -m torch.distributed.launch --nproc_per_node=8 main_pretrain_rsp_llm_global.py \
    --batch_size 16 \
    --accum_iter 12 \
    --model rsp_vit_small_patch16 \
    --epochs 400 \
    --warmup_epochs 40 \
    --video_root /home/byungjun/kinetics400/train2 \
    --json_path /home/byungjun/RSP/artifacts/results/frame_analysis_results_complete.json \
    --embeddings_path /home/byungjun/RSP/artifacts/deberta_embeddings.pt \
    --log_dir logs/rsp-llm-global \
    --output_dir outputs/rsp-llm-global \
    --norm_pix_loss \
    --repeated_sampling 2

  python -m torch.distributed.launch --nproc_per_node=8 --master_port 29600 main_pretrain_rsp_llm_global.py \
    --batch_size 12 \
    --accum_iter 16 \
    --model rsp_vit_small_patch16 \
    --epochs 400 \
    --warmup_epochs 40 \
    --video_root /home/junyoon/kinetics400/train2 \
    --json_path /home/junyoon/rsp/RSP/RSP/artifacts/results/frame_analysis_results_complete.json \
    --embeddings_path /home/junyoon/rsp/RSP/RSP/artifacts/deberta_embeddings.pt \
    --log_dir logs/rsp-llm-global \
    --output_dir outputs/rsp-llm-global \
    --norm_pix_loss \
    --repeated_sampling 2


    --output_dir ./output \
    --batch_size 64 \
    --repeated_sampling 2

DEVICE=5
OUTPUT_DIR=/home/junyoon/RSP/outputs/mse-kl_scall0.005_2024-12-28_23-40-38
EPOCH=100

CUDA_VISIBLE_DEVICES=$DEVICE python eval/DAVIS/eval_video_segmentation_davis.py \
  --finetune $OUTPUT_DIR/checkpoint-$EPOCH.pth \
  --output_dir  $OUTPUT_DIR/davis_seg$EPOCH \
  --data_path /data/DAVIS_480_880 \
  --topk 7 --size_mask_neighborhood 30 --n_last_frames 30 \
  --model vit_small

git clone https://github.com/davisvideochallenge/davis2017-evaluation
CUDA_VISIBLE_DEVICES=3 python ./davis2017-evaluation/evaluation_method.py \
  --task semi-supervised \
  --results_path  $OUTPUT_DIR/davis_seg$EPOCH \
  --davis_path /data/DAVIS_480_880

python -m torch.distributed.launch --nproc_per_node=8 --use_env --master_port 30200 main_pretrain_combined.py -cn mse exp_name=mse-kl_scall0.001 batch_size=48 accum_iter=4 model_params.kl_scale=0.001
python -m torch.distributed.launch --nproc_per_node=4 --use_env --master_port 30100 main_pretrain_combined.py -cn mse exp_name=full-grad_clip batch_size=96 accum_iter=4 