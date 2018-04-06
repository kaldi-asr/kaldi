#!/bin/bash 

. ./cmd.sh

. ./path.sh
stage=0
. ./utils/parse_options.sh

set -e
set -o pipefail
set -u

# The corpus and lexicon are on openslr.org
lexicon="http://www.openslr.org/resources/34/santiago.tar.gz"
speech="http://www.openslr.org/resources/39/LDC2006S37.tar.gz"

tmpdir=data/local/tmp

# where to put the downloaded speech corpus
download_dir=$tmpdir/speech
data_dir=$download_dir/LDC2006S37/data
# If you already have the corpus, put the path  here
datadir=$data_dir

# location of subs text data
subsdata="http://opus.lingfil.uu.se/download.php?f=OpenSubtitles2016/en-es.txt.zip"

if [ $stage -le 0 ]; then
  mkdir -p $tmpdir/heroico $tmpdir/usma $download_dir

  # download the corpus from openslr
  if [ ! -f $download_dir/LDC2006S37.tar.gz ]; then
    wget -O $download_dir/LDC2006S37.tar.gz $speech

  (
    #run in shell, so we don't have to remember the path
    cd $download_dir
    tar -xzf LDC2006S37.tar.gz
  )
  local/prepare_data.sh $data_dir
  else
    local/prepare_data.sh $datadir
  fi
fi

if [ $stage -le 1 ]; then
  mkdir -p data/local/dict $tmpdir/dict

  # download the dictionary from openslr
  if [ ! -f $tmpdir/dict/santiago.tar.gz ]; then
    wget -O $tmpdir/dict/santiago.tar.gz $lexicon
  fi
fi

if [ $stage -le 2 ]; then
  (
    cd $tmpdir/dict
    tar -xzf santiago.tar.gz
  )

  local/prepare_dict.sh
fi

if [ $stage -le 3 ]; then
  utils/prepare_lang.sh data/local/dict "<UNK>" data/local/lang data/local/lang
fi

if [ $stage -le 4 ]; then
  # use am training text to train lm
  mkdir -p $tmpdir/heroico/lm

  # get the text from data/train/text
  cut -d " " -f 2- data/train/text > $tmpdir/heroico/lm/train.txt

  # build lm
  local/prepare_lm.sh $tmpdir/heroico/lm/train.txt

  # make the grammar fst
  utils/format_lm.sh \
    data/local/lang data/local/lm/threegram.arpa.gz data/local/dict/lexicon.txt \
    data/lang

fi

if [ $stage -le 5 ]; then
  # extract acoustic features
  mkdir -p exp

  for fld in native nonnative test devtest train; do
    if [ -e data/$fld/cmvn.scp ]; then
      rm data/$fld/cmvn.scp
    fi

    steps/make_plp_pitch.sh \
      --cmd "$train_cmd" --nj 4 data/$fld exp/make_plp_pitch/$fld plp_pitch

    utils/fix_data_dir.sh data/$fld

    steps/compute_cmvn_stats.sh data/$fld exp/make_plp_pitch plp_pitch

    utils/fix_data_dir.sh data/$fld
  done
fi

if [ $stage -le 6 ]; then
  echo "$0 monophone training"
  steps/train_mono.sh --nj 4 --cmd "$train_cmd" data/train data/lang exp/mono
fi

if [ $stage -le 7 ]; then
  (
    # make decoding graph for monophones
    utils/mkgraph.sh data/lang exp/mono exp/mono/graph

  # test monophones
    for x in native nonnative devtest test; do
      steps/decode.sh --nj 8 exp/mono/graph data/$x exp/mono/decode_${x}
    done
  ) &
fi

if [ $stage -le 8 ]; then
  # align with monophones
  steps/align_si.sh \
    --nj 8 --cmd "$train_cmd" data/train data/lang exp/mono exp/mono_ali
fi

if [ $stage -le 9 ]; then
  echo "$0 Starting  triphone training in exp/tri1"
  steps/train_deltas.sh \
    --cmd "$train_cmd" \
    --cluster-thresh 100 \
    1500 25000 \
    data/train data/lang exp/mono_ali exp/tri1 || exit 1;
fi

if [ $stage -le 10 ]; then
  # test cd gmm hmm models
  # make decoding graphs for tri1
  (
    utils/mkgraph.sh data/lang exp/tri1 exp/tri1/graph

    # decode test data with tri1 models
    for x in native nonnative devtest test; do
      steps/decode.sh --nj 8 exp/tri1/graph data/$x exp/tri1/decode_${x}
    done
  ) &
fi

if [ $stage -le 11 ]; then
  # align with triphones
  steps/align_si.sh \
    --nj 8 --cmd "$train_cmd" data/train data/lang exp/tri1 exp/tri1_ali
fi

if [ $stage -le 12 ]; then
  echo "$0 Starting (lda_mllt) triphone training in exp/tri2b"
  steps/train_lda_mllt.sh \
    --splice-opts "--left-context=3 --right-context=3" \
    2000 30000 \
    data/train data/lang exp/tri1_ali exp/tri2b
fi

if [ $stage -le 13 ]; then
  (
    #  make decoding FSTs for tri2b models
    utils/mkgraph.sh data/lang exp/tri2b exp/tri2b/graph

    # decode  test with tri2b models
    for x in native nonnative devtest test; do
      steps/decode.sh --nj 8  exp/tri2b/graph data/$x exp/tri2b/decode_${x}
    done
  ) &
fi

if [ $stage -le 14 ]; then
  # align with lda and mllt adapted triphones
  steps/align_si.sh \
    --use-graphs true --nj 8 --cmd "$train_cmd" data/train data/lang exp/tri2b \
    exp/tri2b_ali
fi

if [ $stage -le 15 ]; then
  echo "$0 Starting (SAT) triphone training in exp/tri3b"
  steps/train_sat.sh \
    --cmd "$train_cmd" 3100 50000 data/train data/lang exp/tri2b_ali exp/tri3b
fi

if [ $stage -le 16 ]; then
  echo "$0 Starting exp/tri3b_ali"
  steps/align_fmllr.sh \
    --nj 8 --cmd "$train_cmd" \
    data/train data/lang exp/tri3b exp/tri3b_ali
fi

if [ $stage -le 17 ]; then
  (
  # make decoding graphs for SAT models
    utils/mkgraph.sh data/lang exp/tri3b exp/tri3b/graph

    # decode test sets with tri3b models
    for x in native nonnative devtest test; do
      steps/decode_fmllr.sh \
      --nj 8 --cmd "$decode_cmd" exp/tri3b/graph data/$x exp/tri3b/decode_${x}
    done
  ) &
fi

if [ $stage -le 18 ]; then
  # train and test chain models
  local/chain/run_tdnn.sh
fi

wait
