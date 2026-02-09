export PYTHONPATH=/mnt/amlfs-01/shared/shenyuang/DreamDojo-release:$PYTHONPATH
export HF_HOME=/mnt/amlfs-01/shared/shenyuang/cosmos_cache
source /mnt/amlfs-01/shared/shenyuang/DreamDojo-release/.venv/bin/activate

python examples/action_conditioned.py \
  -o outputs/action_conditioned/basic \
  --checkpoints-dir /mnt/amlfs-01/shared/shenyuang/cosmos_logs/exp1201/gr1/checkpoints \
  --experiment groot_ac_reason_embeddings_rectified_flow_2b_480_640_gr1 \
  --save-dir /mnt/amlfs-01/shared/shenyuang/dreamdojo_results/gr1_unified_test \
  --num-frames 49 \
  --num-samples 100 \
  --dataset-path /mnt/amlfs-03/shared/datasets/gr1/gr1_unified_v1/0308/gr1_unified.RU0226RemoveStaticFreq20 \
  --data-split test \
  --deterministic-uniform-sampling \
  --checkpoint-interval 5000 \
  --infinite