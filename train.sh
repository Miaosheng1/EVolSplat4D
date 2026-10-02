#!/usr/bin/env bash

get_free_gpu() {
    # Select the GPU with the least used memory.
    free_gpu=$(nvidia-smi --query-gpu=memory.used,index --format=csv,noheader,nounits | \
        sort -n | \
        awk 'NR==1{print $2}')
    echo $free_gpu
}

FREE_GPU=$(get_free_gpu)
CUDA_VISIBLE_DEVICES=$FREE_GPU ns-train evolsplat4d \
--descriptor unified_depth_input_rigid \
--max_num_iterations 50000 \
--steps_per_eval_image 2000 \
--steps_per_save 10000 \
--config_file config/evolsplat4d_dynamic.yaml \
--data /nas/datasets/DynSplat_unified/unifed_training/
