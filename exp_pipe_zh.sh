#!/bin/bash
#SBATCH -p batch
#SBATCH -t 3-00:00:00
#SBATCH -N 1
#SBATCH -c 6
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1

set -o nounset
set -o errexit
set -o pipefail


{
    . scripts/set_environment.sh
    args=( "$@" )
    for arg in "${args[@]}"; do
        eval "$arg"
    done

    echo "devices:       ${devices:=0,1}"
    echo "seed:          ${seed:=0}"
    echo "error schema:  ${error_schema:=last}"

    exper_path=exp/zhbart.$seed
    # Train GEC model
    bash train_zh.sh  \
        encoder=bart  \
        config=configs/bart.zh.ini  \
        devices=$devices  \
        seed=$seed  \
        update=6  \
        path=$exper_path/model

    # Use GEC model to generate pseudo data for target GED
    bash scripts/generate_chn_treebank/generate_neg_treebank.sh  \
        -d "$devices"  \
        -b 3000  \
        -G  \
        --gec-path "$exper_path"/model.1  \
        --config configs/bart.zh.ini  \
        --gec-topk 12  \
        --output-path "$exper_path"/pseudo_data/  \
        data/ctb7 data/cgec/hsk.train

    # Train target GED model
    bash train_ged_zh.sh  \
        devices="$devices"  \
        config=configs/seq2seq.ged.zh.ini  \
        error_schema="$error_schema"  \
        update=6  \
        data="$exper_path"/pseudo_data  \
        ckptpath="$exper_path"/model.1  \
        path="$exper_path"/ged."$error_schema"/model
}