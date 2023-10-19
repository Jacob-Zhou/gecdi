#!/bin/bash
#SBATCH -p batch
#SBATCH -t 3-00:00:00
#SBATCH -N 1
#SBATCH -c 6
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1

# nohup bash train.sh devices=4,5,6,7 update=5 path=exp/transformer/model encoder=transformer config=configs/transformer.ini > exp/transformer/log 2>&1 &
# nohup bash train.sh devices=4,5,6,7 update=5 path=exp/bart/model        encoder=bart        config=configs/bart.ini        > log.bart 2>&1 &

set -o nounset
set -o errexit
set -o pipefail

{
    . scripts/set_environment.sh
    args=$@
    for arg in $args; do
        eval "$arg"
    done

    echo "devices:   ${devices:=0,1}"
    echo "update:    ${update:=6}"
    echo "seed:      ${seed:=0}"
    echo "path:      ${path:=exp/transformer/model}"
    echo "encoder:   ${encoder:=transformer}"
    echo "config:    ${config:=configs/transformer.ini}"

    printf "Current commits:\n$(git log -1 --oneline)\n3rd parties:\n"
    cd 3rdparty/parser/ && printf "parser\n$(git log -1 --oneline)\n" && cd ../..

    exp_dir="$(dirname "$path")"
    mkdir -p $exp_dir
    mkdir -p $exp_dir/data

    python -u seq2seq.py train -b -d $devices --seed=$seed --update-steps=$update -c $config -p $path.1 --cache --amp --encoder $encoder --bart "fnlp/bart-large-chinese" --train data/cgec/lang8_5xhsk.yuezhang.train --dev data/cgec/mucgec.dev
}