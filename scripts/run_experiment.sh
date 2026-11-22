#!/usr/bin/env bash
set -euo pipefail

# Usage:
# bash scripts/run_experiment.sh <dataset> <backbone> [seed] [gpu]
#
# Examples:
# bash scripts/run_experiment.sh mvtec mnv3 0 0
# bash scripts/run_experiment.sh visa wr50 0 0

DATASET="${1:?Specify mvtec or visa}"
BACKBONE="${2:?Specify mnv3 or wr50}"
SEED="${3:-0}"
GPU="${4:-0}"

cd "$(dirname "$0")/.."

case "$DATASET" in
    mvtec)
        DATA_ROOT="${MVTEC_ROOT:-./data/mvtec}"
        DATASET_NAME="mvtec"
        CLASSES=(
            bottle cable capsule carpet grid hazelnut leather
            metal_nut pill screw tile toothbrush transistor wood zipper
        )
        ;;
    visa)
        DATA_ROOT="${VISA_ROOT:-./data/visa_pytorch/1cls}"
        DATASET_NAME="visa"
        CLASSES=(
            candle capsules cashew chewinggum fryum macaroni1
            macaroni2 pcb1 pcb2 pcb3 pcb4 pipe_fryum
        )
        ;;
    *)
        echo "Unsupported dataset: $DATASET"
        exit 1
        ;;
esac

case "$BACKBONE" in
    mnv3)
        MODEL_NAME="mobilenetv3_small"
        LAYERS=(features.3 features.8)
        EMBED_DIM=128
        ;;
    wr50)
        MODEL_NAME="wideresnet50"
        LAYERS=(layer2 layer3)
        EMBED_DIM=1024
        ;;
    *)
        echo "Unsupported backbone: $BACKBONE"
        exit 1
        ;;
esac

DATASET_FLAGS=()
for cls in "${CLASSES[@]}"; do
    DATASET_FLAGS+=(-d "$cls")
done

LAYER_FLAGS=()
for layer in "${LAYERS[@]}"; do
    LAYER_FLAGS+=(-le "$layer")
done

GROUP="${DATASET}_${BACKBONE}_seed${SEED}"

python bin/run_patchcore.py \
    --gpu "$GPU" \
    --seed "$SEED" \
    --save_patchcore_model \
    --save_segmentation_images \
    --log_group "$GROUP" \
    --log_project "${DATASET}_Results" \
    results \
    patch_core \
    -b "$MODEL_NAME" \
    "${LAYER_FLAGS[@]}" \
    --pretrain_embed_dimension "$EMBED_DIM" \
    --target_embed_dimension "$EMBED_DIM" \
    --anomaly_scorer_num_nn 1 \
    --patchsize 3 \
    sampler -p 0.1 approx_greedy_coreset \
    dataset \
    "${DATASET_FLAGS[@]}" \
    "$DATASET_NAME" "$DATA_ROOT"
