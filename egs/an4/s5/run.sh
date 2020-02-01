#!/usr/bin/env bash

# Copyright 2016  Allen Guo

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.

. ./cmd.sh
. ./path.sh
set -e  # exit on error

stage=0
nj=1
data=/home/allen/data
an4_root=$data/an4
data_url=http://www.speech.cs.cmu.edu/databases/an4/

# download data (if necessary)
if [ $stage -le 0 ]; then
    local/download_and_untar.sh $data $data_url
fi

# data prep
if [ $stage -le 1 ]; then
    mkdir -p data/{train,test} exp

    if [ ! -f $an4_root/README ]; then
        echo Cannot find an4 root! Exiting...
        exit 1
    fi

    python local/data_prep.py $an4_root $KALDI_ROOT/tools/sph2pipe_v2.5/sph2pipe   

    for x in test train; do
        for f in text wav.scp utt2spk; do
            sort data/$x/$f -o data/$x/$f
        done
        utils/utt2spk_to_spk2utt.pl data/$x/utt2spk > data/$x/spk2utt
    done
fi

# MFCC feature extraction
if [ $stage -le 2 ]; then
    mfccdir='mfcc'
    for x in test train; do
        steps/make_mfcc.sh --nj $nj --cmd "$train_cmd" \
            data/$x exp/make_mfcc/$x $mfccdir
        steps/compute_cmvn_stats.sh data/$x exp/make_mfcc/$x $mfccdir
        utils/fix_data_dir.sh data/$x
    done
fi

# lang prep
if [ $stage -le 3 ]; then
    rm -rf data/local/{dict,lang}
    mkdir -p data/local/{dict,lang}

    python local/lexicon_prep.py $an4_root 
    echo '<UNK> SIL' >> data/local/dict/lexicon.txt
    cat $an4_root/etc/an4.phone | grep -v 'SIL' > data/local/dict/nonsilence_phones.txt
    echo 'SIL' > data/local/dict/silence_phones.txt
    echo 'SIL' > data/local/dict/optional_silence.txt
    echo -n > data/local/dict/extra_questions.txt

    utils/prepare_lang.sh data/local/dict "<UNK>" data/local/lang data/lang
    arpa2fst $an4_root/etc/an4.ug.lm | fstprint | \
        grep -v 'HALL\|LANE\|MEMORY\|TWELVTH\|WEAN' | \
        utils/remove_oovs.pl data/lang/oov.txt | utils/eps2disambig.pl | utils/s2eps.pl | \
        fstcompile --isymbols=data/lang/words.txt --osymbols=data/lang/words.txt --keep_isymbols=false --keep_osymbols=false | \
        fstarcsort > data/lang/G.fst
fi

# train monophone system 
if [ $stage -le 4 ]; then
    steps/train_mono.sh --nj $nj --cmd "$train_cmd" data/train data/lang exp/mono
    utils/mkgraph.sh data/lang exp/mono exp/mono/graph
    steps/decode.sh --config conf/decode.config --nj $nj --cmd "$decode_cmd" \
        exp/mono/graph data/test exp/mono/decode
fi

# align mono
if [ $stage -le 5 ]; then
    steps/align_si.sh --nj $nj --cmd "$train_cmd" \
        data/train data/lang exp/mono exp/mono_ali
fi

# train tri1 (first triphone pass) and decode
if [ $stage -le 6 ]; then
    steps/train_deltas.sh --cmd "$train_cmd" \
        1800 9000 data/train data/lang exp/mono_ali exp/tri1
    utils/mkgraph.sh data/lang exp/tri1 exp/tri1/graph
    steps/decode.sh --config conf/decode.config --nj $nj --cmd "$decode_cmd" \
        exp/tri1/graph data/test exp/tri1/decode
fi

# align tri1 
if [ $stage -le 7 ]; then
    steps/align_si.sh --nj $nj --cmd "$train_cmd" \
        data/train data/lang exp/tri1 exp/tri1_ali
fi

# train tri2 (LDA+MLLT) and decode
if [ $stage -le 8 ]; then
    steps/train_lda_mllt.sh --cmd "$train_cmd" \
        1800 9000 data/train data/lang exp/tri1_ali exp/tri2
    utils/mkgraph.sh data/lang exp/tri2 exp/tri2/graph
    steps/decode.sh --config conf/decode.config --nj $nj --cmd "$decode_cmd" \
        exp/tri2/graph data/test exp/tri2/decode
fi

# align tri2
if [ $stage -le 9 ]; then
    steps/align_si.sh --nj $nj --cmd "$train_cmd" \
        data/train data/lang exp/tri2 exp/tri2_ali
fi

# train tri3 (LDA+MLLT+SAT) and decode
if [ $stage -le 10 ]; then
    steps/train_sat.sh 1800 9000 data/train data/lang exp/tri2_ali exp/tri3
    utils/mkgraph.sh data/lang exp/tri3 exp/tri3/graph
    steps/decode_fmllr.sh --config conf/decode.config --nj $nj --cmd "$decode_cmd" \
        exp/tri3/graph data/test exp/tri3/decode
fi
