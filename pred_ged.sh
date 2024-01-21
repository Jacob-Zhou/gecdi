#!/bin/bash
#SBATCH -p batch
#SBATCH -t 3-00:00:00
#SBATCH -N 1
#SBATCH -c 6
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
    echo "path:         ${path:=exp/transformer/model}"
    echo "data:         ${data:=}"
    echo "config:       ${config:=configs/bart.ini}"
    echo "pred:         ${pred:=$(dirname "$path")/pred/ged.pred.jsonl}"

    printf "Current commits:\n%s\n3rd parties:\n" "$(git log -1 --oneline)"
    cd 3rdparty/parser/ && printf "parser\n%s\n" "$(git log -1 --oneline)" && cd ../..

    mkdir -p "$(dirname "$pred")"

    python -u seq2seq_ged.py predict --cache --amp  \
        -d "$devices"  \
        -c "$config"  \
        -p "$path"  \
        --data "$data"  \
        --pred "$pred"
}