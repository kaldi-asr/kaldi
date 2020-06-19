#!/usr/bin/env bash

set -e
stage=0
nj=60

database_train=/export/corpora5/handwriting_ocr/CASIA_HWDB/Offline/
database_competition=/export/corpora5/handwriting_ocr/CASIA_HWDB/Offline/
data_dir=data
exp_dir=exp

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

if [ $stage -le -1 ]; then
    mkdir download/Train
    mkdir download/Test
    mkdir download/Competition
    local/extract_database.sh --database-train $database_train \
        --database-competition $database_competition
fi

if [ $stage -le 0 ]; then
    mkdir -p data/train/data/images
    mkdir -p data/test/data/images
    mkdir -p data/competition/data/images
    local/process_data.py download/Train data/train
    local/process_data.py download/Test data/test
    local/process_data.py download/Competition data/competition
    image/fix_data_dir.sh ${data_dir}/test
    image/fix_data_dir.sh ${data_dir}/train
    image/fix_data_dir.sh ${data_dir}/competition
fi

mkdir -p $data_dir/{train,test}/data
if [ $stage -le 1 ]; then
    echo "$0: Obtaining image groups. calling get_image2num_frames"
    echo "Date: $(date)."
    image/get_image2num_frames.py --feat-dim 60 $data_dir/train
    image/get_allowed_lengths.py --frame-subsampling-factor 4 10 $data_dir/train

    for datasplit in train test competition; do
        echo "$0: Extracting features and calling compute_cmvn_stats for dataset: $datasplit. "
        echo "Date: $(date)."
        local/extract_features.sh --nj $nj --cmd "$cmd" \
            --feat-dim 60 --num-channels 3 \
            $data_dir/${datasplit}
        steps/compute_cmvn_stats.sh $data_dir/${datasplit} || exit 1;
    done

    echo "$0: Fixing data directory for train dataset"
    echo "Date: $(date)."
    utils/fix_data_dir.sh $data_dir/train
fi

#if [ $stage -le 2 ]; then
#    for datasplit in train; do
#        echo "$(date) stage 2: Performing augmentation, it will double training data"
#        local/augment_data.sh --nj $nj --cmd "$cmd" --feat-dim 60 $data_dir/${datasplit} $data_dir/${datasplit}_aug $data_dir
#        steps/compute_cmvn_stats.sh $data_dir/${datasplit}_aug || exit 1;
#    done
#fi

if [ $stage -le 3 ]; then
    echo "$0: Preparing dictionary and lang..."
    if [ ! -f $data_dir/train/bpe.out ]; then
        cut -d' ' -f2- $data_dir/train/text | utils/lang/bpe/prepend_words.py | python3 utils/lang/bpe/learn_bpe.py -s 700 > $data_dir/train/bpe.out
        for datasplit in test train; do
            cut -d' ' -f1 $data_dir/$datasplit/text > $data_dir/$datasplit/ids
            cut -d' ' -f2- $data_dir/$datasplit/text | utils/lang/bpe/prepend_words.py | python3 utils/lang/bpe/apply_bpe.py -c $data_dir/train/bpe.out | sed 's/@@//g' > $data_dir/$datasplit/bpe_text
            mv $data_dir/$datasplit/text $data_dir/$datasplit/text.old
            paste -d' ' $data_dir/$datasplit/ids $data_dir/$datasplit/bpe_text > $data_dir/$datasplit/text
        done
    fi

    local/prepare_dict.sh --data-dir $data_dir --dir $data_dir/local/dict
    # This recipe uses byte-pair encoding, the silences are part of the words' pronunciations.
    # So we set --sil-prob to 0.0
    utils/prepare_lang.sh --num-sil-states 4 --num-nonsil-states 8 --sil-prob 0.0 --position-dependent-phones false \
        $data_dir/local/dict "<sil>" $data_dir/lang/temp $data_dir/lang
    utils/lang/bpe/add_final_optional_silence.sh --final-sil-prob 0.5 $data_dir/lang
fi

if [ $stage -le 4 ]; then
    echo "$0: Estimating a language model for decoding..."
    local/train_lm.sh --data-dir $data_dir  --dir $data_dir/local/local_lm
    utils/format_lm.sh $data_dir/lang $data_dir/local/local_lm/data/arpa/3gram_unpruned.arpa.gz \
        $data_dir/local/dict/lexicon.txt $data_dir/lang_test
fi

if [ $stage -le 5 ]; then
    echo "$0: Calling the flat-start chain recipe..."
    echo "Date: $(date)." 
    local/chain/run_flatstart_cnn1a.sh --nj $nj --train-set train --data-dir $data_dir --exp-dir $exp_dir
fi

if [ $stage -le 6 ]; then
    echo "$0: Aligning the training data using the e2e chain model..."
    echo "Date: $(date)."
    steps/nnet3/align.sh --nj $nj --cmd "$cmd" --use-gpu false \
        --scale-opts '--transition-scale=1.0 --acoustic-scale=1.0 --self-loop-scale=1.0' \
        $data_dir/train $data_dir/lang $exp_dir/chain/e2e_cnn_1a $exp_dir/chain/e2e_ali_train
fi

if [ $stage -le 7 ]; then
    echo "$0: Building a tree and training a regular chain model using the e2e alignments..."
    echo "Date: $(date)."
    local/chain/run_cnn_e2eali_1b.sh --nj $nj --train-set train --data-dir $data_dir --exp-dir $exp_dir
fi
