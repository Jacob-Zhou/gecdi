#!/bin/bash

set -o nounset
set -o errexit
set -o pipefail

check_python_package() {
  if python -c "import pkgutil; exit(pkgutil.find_loader('$1') is not None)"; then
    echo "- '$1' package is not installed. Installing it now..."
    if pip install "$1"; then
      echo "- '$1' package installed successfully."
    else
      echo "Error: failed to install '$1' package." >&2
      exit 1
    fi
  else
    echo "- '$1' package is already installed."
  fi
}

show_help() {
    echo "Usage: $(basename "$0") [-h] [-d devices] [-b batch-size] -G --gec-path gec-path [--output-path output-path] parallel_files"
    echo "   -h                Display help information"
    echo "   -G                Wheather to generate topk negative"
    echo "   -s                Random seed"
    echo "   parallel_files"
}

{
    for arg in "$@"; do
    shift
    case "$arg" in
        '--help')   set -- "$@" '-h'   ;;
        '--devices')   set -- "$@" '-d'   ;;
        '--config')   set -- "$@" '-c'   ;;
        '--batch')   set -- "$@" '-b'   ;;
        '--generate-topk-neg') set -- "$@" '-G'   ;;
        '--gec-path')   set -- "$@" '-g'   ;;
        '--gec-topk')   set -- "$@" '-k'   ;;
        '--output-path')   set -- "$@" '-o'   ;;
        '--seed')   set -- "$@" '-s'   ;;
        *)          set -- "$@" "$arg" ;;
    esac
    done

    # Default behavior
    generate_topk_neg=false
    devices='0,1,2,3,4,5,6,7'
    batch=5000
    topk=5
    seed=1
    gec_path=''
    output_path='output'
    parallel_files=()
    portions=("train" "dev")

    # Parse short options
    OPTIND=1
    while getopts ":hd:c:b:Gg:k:o:s:" opt
    do
    case "$opt" in
        h) show_help && exit 0 ;;
        d) devices=$OPTARG ;;
        c) config=$OPTARG ;;
        b) batch=$OPTARG ;;
        G) generate_topk_neg=true ;;
        k) topk=$OPTARG ;;
        g) gec_path=$OPTARG ;;
        o) output_path=$OPTARG ;;
        s) seed=$OPTARG ;;
        \?) echo "Invalid option: -$OPTARG" >&2 && exit 1 ;;
        :) echo "Option -$OPTARG requires an argument." >&2 && exit 1 ;;
    esac
    done
    shift "$(( OPTIND - 1 ))" # remove options from positional parameters

    if $generate_topk_neg ; then
        if [ -z "$gec_path" ]; then
            echo "Error: Expect gec_path for generate topk negative example" >&2
            exit 1
        fi
        if [ ! -e "$gec_path" ]; then
            echo "Error: gec_path '$gec_path' does not exist." >&2
            exit 1
        fi
    fi

    # Process non-option arguments
    if [ $# -lt 1 ]; then
        echo "Error: Expected at least one non-option argument (parallel_files)" >&2
        show_help
        exit 1
    fi

    has_error=false
    # shift
    while [ $# -gt 0 ]; do
        file=$1
        if [ -e "$file" ]; then
            parallel_files+=("$file")
        else
            echo "Error: file '$file' does not exist." >&2
            has_error=true
        fi
        shift
    done
    if $has_error; then
        exit 1
    fi

    echo "Args:"
    echo "- generate-topk-neg:      $generate_topk_neg"
    if $generate_topk_neg; then
        echo "- gec-path:               $gec_path"
        echo "- gec-config:             $config"
    fi
    echo "- parallel_files:"
    for file in "${parallel_files[@]}"; do
        echo "  + $file"
    done
    echo "- output_path:            $output_path"
    echo

    # check requirements
    echo "Check requirements:"
    check_python_package "supar"

    mkdir -p "$output_path"

    tmp_path="$output_path"/tmp
    mkdir -p "$tmp_path"

    # Step 0. generate topk negative sample
    echo
    echo "Generate topk negative sample"
    echo "  - config:  ${config:=configs/transformer.ini}"
    echo "  - devices: ${devices:=0,1,2,3}"
    echo "  - batch:   ${batch:=10000}"
    echo "  - topk:    ${topk:=5}"
    for file in "${parallel_files[@]}"; do
        echo "basename $file"
        python scripts/generate_chn_treebank/discard_correct.py --input-file "$file" --output-file "$tmp_path"/"$(basename "$file")".neg
        python -u seq2seq.py predict -d "$devices" -c "$config" -p "$gec_path" --data "$tmp_path"/"$(basename "$file")".neg --pred "$tmp_path"/"$(basename "$file")".pred --cache --batch-size="$batch" --beam-size="$topk" --topk="$topk"
        # compose predicted file to original file
        python scripts/generate_chn_treebank/compose_predict_file.py --gold-file "$tmp_path"/"$(basename "$file")".neg --pred-file "$tmp_path"/"$(basename "$file")".pred --output-file "$tmp_path"/"$(basename "$file")".parallel
    done
    gec_path="$(dirname "$gec_path")"
    if [ -e "${gec_path:?}"/bin ]; then
        echo "Clean parsing caches"
        rm -r "${gec_path:?}"/bin
    fi

    # Step 1 Split parallel data
    ## split it to tgt_file
    echo
    echo "Step 1: Pre-processing files"
    tmp_files=()
    for file in "${parallel_files[@]}"; do
        ## P: predicted, S: golden-soure, T: golden-target
        echo "* Pre-processing $file"
        python scripts/generate_chn_treebank/split_instances.py --seed "$seed" --input-file "$tmp_path"/"$(basename "$file")".parallel --output-file "$tmp_path"/"$(basename "$file")"
        tmp_files+=("$tmp_path"/"$(basename "$file")".p_train "$tmp_path"/"$(basename "$file")".p_dev)
    done
    echo "Pre-processed files:"
    parallel_files=("${tmp_files[@]}")
    for file in "${parallel_files[@]}"; do
        echo "  + $file"
    done

    echo
    echo "Step 3: Convert Grammatical tree to Ungrammatical tree"
    # Step 4. Project the target-side trees to source-side ones
    for file in "${parallel_files[@]}"; do
        echo "* Convert Grammatical tree in $file to Ungrammatical tree"
        python scripts/generate_chn_treebank/expand_correct_lines.py --input-file "$tmp_path"/"$(basename "$file")".correct --match-file "$tmp_path"/"$(basename "$file")".match --output-file "$tmp_path"/"$(basename "$file")".expanded.correct
        # by default we set processes to 8 and chunksize to 64
        python scripts/generate_chn_treebank/convert_tree.py --processes 20 --source-file "$tmp_path"/"$(basename "$file")".error --target-file "$tmp_path"/"$(basename "$file")".expanded.correct --output-file "$tmp_path"/"$(basename "$file")".error.tree --tokenizer "fnlp/bart-large-chinese"
        python scripts/generate_chn_treebank/convert_tree.py --processes 20 --source-file "$tmp_path"/"$(basename "$file")".source --target-file "$tmp_path"/"$(basename "$file")".correct --output-file "$tmp_path"/"$(basename "$file")".source.tree --tokenizer "fnlp/bart-large-chinese"
        python scripts/generate_chn_treebank/convert_tree.py --processes 20 --source-file "$tmp_path"/"$(basename "$file")".correct --target-file "$tmp_path"/"$(basename "$file")".correct --output-file "$tmp_path"/"$(basename "$file")".correct.tree --tokenizer "fnlp/bart-large-chinese"
        python scripts/generate_chn_treebank/expand_correct_lines.py --input-file "$tmp_path"/"$(basename "$file")".source.tree --match-file "$tmp_path"/"$(basename "$file")".match --output-file "$tmp_path"/"$(basename "$file")".expanded.source.tree
        paste "$tmp_path"/"$(basename "$file")".expanded.source.tree "$tmp_path"/"$(basename "$file")".error.tree > "$tmp_path"/"$(basename "$file")".silver.pid
        paste "$tmp_path"/"$(basename "$file")".source.tree "$tmp_path"/"$(basename "$file")".correct.tree >> "$tmp_path"/"$(basename "$file")".silver.pid
    done

    # Step 5. Merge Files
    ## Ungrammatical tree: *.error.tree
    ## Grammatical tree:   *.correct.tree.pred
    # 1) compose all silver data
    # 2) random split into silver-train, silver-dev and silver-test,
    # 3) compose silver with gold
    echo
    echo "Step 4: Merging data and split into {train, dev, test}"

    for portion in "${portions[@]}"; do
        touch "$tmp_path"/silver."$portion".pid
        echo -n > "$tmp_path"/silver."$portion".pid
    done
    for file in "${parallel_files[@]}"; do
        if [[ $(basename "$file") =~ "p_dev" ]]; then
            cat "$tmp_path"/"$(basename "$file")".silver.pid >> "$tmp_path"/silver.dev.pid
        elif [[ $(basename "$file") =~ "p_test" ]]; then
            cat "$tmp_path"/"$(basename "$file")".silver.pid >> "$tmp_path"/silver.test.pid
        else
            cat "$tmp_path"/"$(basename "$file")".silver.pid >> "$tmp_path"/silver.train.pid
        fi
    done
    for portion in "${portions[@]}"; do
        # Filter illegal data in the input file
        python scripts/generate_chn_treebank/filter_illegal.py --input-file "$tmp_path"/silver."$portion".pid --output-file "$output_path"/"$portion".pid
    done

    # Clean
    rm -rf "$tmp_path"
}