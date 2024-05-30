#!/usr/bin/env bash

# Copyright 2020  Johns Hopkins University (author: Piotr Żelasko)
# Apache 2.0

stage=0

. path.sh
. utils/parse_options.sh

# Pre-requisites
pip install lhotse
pip install git+https://github.com/pzelasko/Praat-textgrids

# Use Lhotse to prepare the data dir
if [ $stage -le 0 ]; then
    lhotse prepare nsc /export/corpora5/nsc data/nsc
    lhotse kaldi export data/nsc/recordings_PART3_SameCloseMic.json data/nsc/supervisions_PART3_SameCloseMic.json data/nsc
    utils/fix_data_dir.sh data/nsc
    utils/utt2spk_to_spk2utt.pl data/nsc/utt2spk > data/nsc/spk2utt
    # "Poor man's text normalization"
    mv data/nsc/text data/nsc/text.bak
    cat data/nsc/text.bak \
        | sed 's/[#!~()*]\+//g' \
        | sed 's/<UNK>/XPLACEHOLDERX/g' \
        | sed 's/<.\+>//g' \
        | sed 's/XPLACEHOLDERX/<UNK>/g' \
        > data/nsc/text

    # Create a train and test split
    n_spk=$(wc -l data/nsc/spk2utt | cut -f1 -d' ')
    tail -10 data/nsc/spk2utt | cut -f1 -d' ' > data/test.spk
    head -n $((n_spk - 10)) data/nsc/spk2utt | cut -f1 -d' ' > data/train.spk
    utils/subset_data_dir.sh --spk-list data/train.spk data/nsc data/train
    utils/subset_data_dir.sh --spk-list data/test.spk data/nsc data/test
fi

# Prepare the dict dir
if [ $stage -le 1 ]; then
    lexicon_path=/export/corpora5/nsc/LEXICON/LEXICON.txt
    mkdir -p data/local/dict
    # Lexicon
    echo "<SIL> <SIL>" > data/local/dict/lexicon.txt
    echo "<UNK> <UNK>" >> data/local/dict/lexicon.txt
    # We are removing a couple of broken entries from the NSC lexicon
    # (I found them with trial-and-error)
    cut '-f-2' $lexicon_path | sort | uniq \
        | grep -v '^s ai l O n' \
        | grep -v '^M OW HH AA NN' \
        | grep -v '^M UW AA M UW S AA NG' \
        | grep -v '^ts\\ i n h au' \
        >> data/local/dict/lexicon.txt
    # Nonsilence phones
    cut '-f2' $lexicon_path | sed 's/ /\n/g' | sort | uniq > data/local/dict/nonsilence_phones.txt
    # Silence phones
    echo "<SIL>" > data/local/dict/silence_phones.txt
    echo "<UNK>" >> data/local/dict/silence_phones.txt
    # Extra questions
    touch data/local/dict/extra_questions.txt
    # Optional silence
    echo "<SIL>" > data/local/dict/optional_silence.txt
fi

# Prepare the lang dir
if [ $stage -le 2 ]; then
    utils/prepare_lang.sh data/local/dict '<UNK>' data/local/lang data/lang
fi

# Prepare the LM
if [ $stage -le 3 ]; then
    dir=data/local/lm
    mkdir $dir
    # Train/Dev split + vocabulary
    heldout=10000
    cut -f2- -d' ' data/train/text > $dir/train.all.txt
    tail -n +10001 $dir/train.all.txt > $dir/train.txt
    head -n +10000 $dir/train.all.txt > $dir/dev.txt
    cut -d' ' -f1 data/local/dict/lexicon.txt  > $dir/wordlist

    ngram-count -text $dir/train.txt -order 3 -limit-vocab -vocab $dir/wordlist \
        -unk -map-unk "<UNK>" -kndiscount -interpolate -lm $dir/nsc.o3g.kn.gz
    echo "PPL for NSC trigram LM:"
    ngram -unk -lm $dir/nsc.o3g.kn.gz -ppl $dir/dev.txt | tee $dir/3gram.ppl

    ngram-count -text $dir/train.txt -order 4 -limit-vocab -vocab $dir/wordlist \
        -unk -map-unk "<UNK>" -kndiscount -interpolate -lm $dir/nsc.o4g.kn.gz
    echo "PPL for NSC 4gram LM:"
    ngram -unk -lm $dir/nsc.o4g.kn.gz -ppl $dir/dev.txt | tee $dir/4gram.ppl
fi

# Prepare the lang_test directory
if [ $stage -le 4 ]; then
    utils/format_lm.sh data/lang data/local/lm/nsc.o3g.kn.gz data/local/dict/lexicon.txt data/lang_test_tg
    utils/build_const_arpa_lm.sh data/local/lm/nsc.o4g.kn.gz data/lang/ data/lang_test_fg
fi
