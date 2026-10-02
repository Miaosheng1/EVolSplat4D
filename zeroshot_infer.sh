#!/usr/bin/env bash

get_free_gpu() {
    free_gpu=$(nvidia-smi --query-gpu=memory.used,index --format=csv,noheader,nounits | \
        sort -n | \
        awk 'NR==1{print $2}')
    echo $free_gpu
}

dirs=(
    "scene_121_000_060"
    "scene_130_130_190"
    "scene_140_040_100"
    "scene_146_020_080"
    "scene_146_090_150"
)

data_root=/nas/datasets/DynSplat_unified/unified_testing
config_file=config/evolsplat4d_dynamic.yaml
eval_mode=drop80
case "${1:-dynamic}" in
    dynamic) ;;
    static)
        data_root=/nas/users/smiao/Dynsplat_data_prior/infer_evolsplat4d_static/waymo
        config_file=config/evolsplat4d_static.yaml
        eval_mode=drop50
        dirs=(
            scene_003_039_089
            scene_226_031_081
            scene_245_098_148
            scene_246_050_100
            scene_271_110_160
            scene_297_070_120
        )
        ;;
    *) echo "Usage: bash zeroshot_infer.sh [dynamic|static]" >&2; exit 1 ;;
esac

FREE_GPU=$(get_free_gpu)
for dir in "${dirs[@]}"; do
    echo "Testing on dataset: ${dir}"
    CUDA_VISIBLE_DEVICES=$FREE_GPU python nerfstudio/scripts/infer_4d.py evolsplat4d \
        --load_checkpoint weight/pretrain_waymo.ckpt \
        --config_file "${config_file}" \
        --pipeline.model.freeze_volume=True \
        evolsplat4d-zeroshot-data \
        --data "${data_root}/${dir}" \
        --eval_mode "${eval_mode}"
done
