#!/bin/bash
#SBATCH -p batch
#SBATCH -t 7-00:00:00
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

    echo "devices: ${devices:=0,1}"
    echo "update:  ${update:=5}"
    echo "seed:    ${seed:=0}"
    echo "path:    ${path:=exp/transformer/model}"
    echo "encoder: ${encoder:=transformer}"
    echo "config:  ${config:=configs/transformer.ini}"

    printf "Current commits:\n$(git log -1 --oneline)\n3rd parties:\n"
    cd 3rdparty/parser/ && printf "parser\n$(git log -1 --oneline)\n" && cd ../..

    if [ $encoder = "transformer" ]; then
        lr2=5e-5
        lr3=5e-6
        warm2=2000
        warm3=4000
    else
        lr2=5e-6
        lr3=3e-6
        warm2=0
        warm3=0
    fi

    python -u seq2seq.py train -b -d $devices --seed=$seed --update-steps=$update -c $config -p $path.1 --cache --amp --encoder $encoder --train data/clang8.train --vocab data/train.tgt
    cp $path.1 $path.2
    python -u seq2seq.py train    -d $devices --seed=$seed --update-steps=$update -c $config -p $path.2 --cache --amp --encoder $encoder --train data/error_coded.train --lr=$lr2 --warmup-steps=$warm2
    cp $path.2 $path.3
    python -u seq2seq.py train    -d $devices --seed=$seed --update-steps=$update -c $config -p $path.3 --cache --amp --encoder $encoder --train data/wi_locness.train  --lr=$lr3 --warmup-steps=$warm3
    bash pred.sh devices=$devices path=$path.1 config=$config pred=$(dirname $path)/test.pred.1 dataset=bea19.dev batch=1000
    bash pred.sh devices=$devices path=$path.2 config=$config pred=$(dirname $path)/test.pred.2 dataset=bea19.dev batch=1000
    bash pred.sh devices=$devices path=$path.3 config=$config pred=$(dirname $path)/test.pred.3 dataset=bea19.dev batch=1000
}