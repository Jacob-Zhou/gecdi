#!/bin/bash
#SBATCH -p batch
#SBATCH -t 3-00:00:00
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1

# nohup bash train_syn_wrapper.sh devices=4,5,6,7 update=5 ckptpath=exp/transformer.2.debug/model.3 path=exp/syn_wrapper/model config=configs/transformer.ini > exp/transformer/log 2>&1 &
{
    . scripts/set_environment.sh
    args=( "$@" )
    for arg in "${args[@]}"; do
        eval "$arg"
    done

    echo "devices:      ${devices:=0}"
    echo "update:       ${update:=5}"
    echo "seed:         ${seed:=1}"
    echo "ckptpath:     ${ckptpath:=exp/transformer/model}"
    echo "path:         ${path:=exp/transformer/model}"
    echo "data:         ${data:=syn-data-step-2-3.bart-with-weight-decay.corase.top-12}"
    echo "config:       ${config:=configs/transformer.ini}"
    echo "batchs:       ${batchs:=65536}"
    echo "error_schema: ${error_schema:=partial-last}"
    echo "lr:           ${lr:=5e-6}"
    echo "warm:         ${warm:=0}"
    echo "lr-rate:      ${lr_rate:=5}"

    printf "Current commits:\n%s\n3rd parties:\n" "$(git log -1 --oneline)"
    cd 3rdparty/parser/ && printf "parser\n%s\n" "$(git log -1 --oneline)" && cd ../..

        # --gec-init  \
    python -u seq2seq_ged.py train --cache --amp  \
        -d "$devices"  \
        -c "$config"  \
        -p "$path"  \
        --checkpoint-path "$ckptpath"  \
        --error-schema "$error_schema"  \
        --train $data/train.pid  \
        --dev $data/dev.pid  \
        --max-len 128  \
        --beam-size=1  \
        --lr=$lr  \
        --chunk-size 2000  \
        --lr-rate=$lr_rate  \
        --label-smoothing=0.0  \
        --batch-size="$batchs"  \
        --warmup-steps=$warm  \
        --update-steps="$update"
}

