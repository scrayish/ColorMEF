#!/bin/bash

eval "$(conda shell.bash hook)"
source activate env_pytorch
cd /home/worker/Documents/exposure-fusion
python3 main.py \
-model transformer \
-use_cuda True \
-data_type SAMPLE \
-train_data \
/sample/path/train \
-test_data \
/sample/path/test \
-output_path \
/sample/path/output \
-checkpoint_path \
/sample/path/checkpoint \
-fuse_expos \
dark middle bright \
-epochs \
200 \
-learning_rate \
1e-4 \
-static_scale_coeff \
1.0 \
-chroma_to_luma_coeff \
1.0 \
-high_size \
1250 \
-low_size \
960 \
-decay_interval \
10 \
-decay_ratio \
0.9 \
-epochs_per_eval \
1 \
-epochs_per_save \
0 \
-dimensions \
512 \
-heads \
8 \
-enhance_length \
3 \
-pad \
reflect \
-readout \
identity
