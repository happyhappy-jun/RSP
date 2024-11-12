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
    --batch_size 8 \
    --accum_iter 24 \
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

    
    --output_dir ./output \
    --batch_size 64 \
    --repeated_sampling 2