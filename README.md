<h1 align="center"> Visual Representation Learning with Stochastic Frame Prediction</h1>
<div align="center">
  <a href="https://huiwon-jang.github.io/" target="_blank">Huiwon&nbsp;Jang</a><sup>1</sup> &ensp; <b>&middot;</b> &ensp;
  <a target="_blank">Dongyoung&nbsp;Kim</a><sup>1</sup> &ensp; <b>&middot;</b> &ensp;
  <a href="https://junsu-kim97.github.io/" target="_blank">Junsu&nbsp;Kim</a><sup>1</sup>
  <br>
  <a href="https://alinlab.kaist.ac.kr/shin.html" target="_blank">Jinwoo&nbsp;Shin</a><sup>1</sup>&ensp; <b>&middot;</b> &ensp;
  <a href="https://people.eecs.berkeley.edu/~pabbeel/" target="_blank">Pieter&nbsp;Abbeel</a><sup>2</sup>&ensp; <b>&middot;</b> &ensp;
  <a href="https://younggyo.me/" target="_blank">Younggyo&nbsp;Seo</a><sup>1,3</sup>
  <br>
  <sup>1</sup> KAIST &emsp; <sup>2</sup>UC Berkeley &emsp; <sup>3</sup>Dyson Robot Learning Lab &emsp; <br>
</div>
<h3 align="center">[<a href="https://sites.google.com/view/2024rsp">project page</a>] [<a href="https://openreview.net/forum?id=rI6lxIX0uX">openreview</a>]</h3>

<img width="100%" src="https://github.com/huiwon-jang/RSP/assets/69646951/7ee0066f-f1a5-4db1-84b5-8ccb3862475a"/>

## Dataset

### Kinetics400
```bash
# Replace corrupted videos
python data_preprocessing/move_replacement_videos.py --replacement_dir /data/kinetics400/original_train2/replacement_for_corrupted_k400 --output_base /data/kinetics400/original_train2
```

Dataset generation procedure

```bash
python -m caption2.steps.step1_extract_frames --data_root /data/kinetics400/original_train2 --output_dir /data/kinetics400/frames
python -m caption2.steps.step2_create_requests --frame_info /data/kinetics400/frames/frame_info.json --output_dir /data/RSP/requests  
python -m caption2.steps.step3_process_captions --requests /data/RSP/requests/caption_requests.json --output_dir /data/RSP/captions       
python -m caption2.steps.step4_check_batch_errors --start_batch batch_676466b1154481908b8c98afebfe45e4 --end_batch batch_67647aeb024c8190a203cfb28332480c --output_dir /data/RSP/error_retry --requests_file /data/RSP/requests/caption_requests.json
python -m caption2.steps.step3_process_captions --requests /data/RSP/error_retry/retry_requests.json --output_dir /data/RSP/error_retry_output
python -m caption2.steps.step5_combine_results \
  --original_results /data/RSP/captions/caption_results.json \
  --retry_results /data/RSP/error_retry_output/caption_results.json \
  --requests_file /data/RSP/requests/caption_requests.json \
  --output_dir /data/RSP/captions
python -m caption2.steps.step6_create_embeddings --caption_results /data/RSP/captions/kinetics400_captions.json --output_dir /data/RSP/captions/embeddings
```

We use seed 42 when sampling first 2 pair of frames for each video. We use seed 43 when sampling additional 6 pair of frame for each video. 
