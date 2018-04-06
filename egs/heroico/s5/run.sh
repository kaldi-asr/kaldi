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
  mkdir -p $tmpdir/heroico
  mkdir -p $tmpdir/usma
  mkdir -p $download_dir

  [ -z "$datadir" ] && \
      # download the corpus from openslr
  if [ ! -f $download_dir/LDC2006S37.tar.gz ]; then
    wget -O $download_dir/LDC2006S37.tar.gz $speech
  fi

  (
    #run in shell, so we don't have to remember the path
    cd $download_dir
    tar -xzf LDC2006S37.tar.gz
  )
  local/prepare_data.sh $data_dir
else
  local/prepare_data.sh $datadir
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
    #run in shell, so we don't have to remember the path
    cd $tmpdir/dict
    tar -xzf santiago.tar.gz
  )

  local/prepare_dict.sh
fi
exit
if [ $stage -le 2 ]; then
  # prepare the lang directory
  utils/prepare_lang.sh \
    data/local/dict "<UNK>" \
    data/local/lang data/lang   || exit 1;
  # use am training text to train lm
  mkdir -p $tmpdir/heroico/lm

  # get the text from data/train/text
  cut -d " " -f 2- data/train/text > $tmpdir/heroico/lm/train.txt

  # build lm
  local/prepare_lm.sh $tmpdir/heroico/lm/train.txt

  utils/format_lm.sh \
    data/lang data/local/lm/threegram.arpa.gz data/local/dict/lexicon.txt \
    data/lang_test
fi

if [ $stage -le 5 ]; then
  # extract acoustic features
  mkdir -p exp

  for fld in native nonnative test devtest train; do
    if [ -e data/$fld/cmvn.scp ]; then
      rm data/$fld/cmvn.scp
    fi

    steps/make_plp_pitch.sh --cmd "$train_cmd" --nj 4 \
      data/$fld exp/make_plp_pitch/$fld plp_pitch || exit 1;

    utils/fix_data_dir.sh data/$fld || exit 1;

    steps/compute_cmvn_stats.sh data/$fld exp/make_plp_pitch plp_pitch || exit 1;

    utils/fix_data_dir.sh data/$fld || exit 1;
  done

  echo "$0 monophone training"
  steps/train_mono.sh \
    --nj 4 --cmd "$train_cmd" \
    data/train data/lang exp/mono || exit 1;

  # evaluation
  (
  # make decoding graph for monophones
  utils/mkgraph.sh \
    data/lang_test \
    exp/mono \
    exp/mono/graph || exit 1;

  # test monophones
  for x in native nonnative devtest test; do
    steps/decode.sh --nj 8  \
      exp/mono/graph data/$x exp/mono/decode_${x} || exit 1;
  done
  ) &

  # align with monophones
  steps/align_si.sh \
    --nj 8 --cmd "$train_cmd" \
    data/train data/lang exp/mono exp/mono_ali || exit 1;

  echo "$0 Starting  triphone training in exp/tri1"
  steps/train_deltas.sh \
    --cmd "$train_cmd" \
    --cluster-thresh 100 \
    1500 25000 \
    data/train data/lang exp/mono_ali exp/tri1 || exit 1;

  # test cd gmm hmm models
  # make decoding graphs for tri1
  (
  utils/mkgraph.sh \
    data/lang_test exp/tri1 exp/tri1/graph || exit 1;

  # decode test data with tri1 models
  for x in native nonnative devtest test; do
    steps/decode.sh \
      --nj 8  \
      exp/tri1/graph data/$x exp/tri1/decode_${x} || exit 1;
  done
  ) &

  # align with triphones
  steps/align_si.sh \
    --nj 8 --cmd "$train_cmd" \
    data/train data/lang exp/tri1 exp/tri1_ali
fi

if [ $stage -le 7 ]; then
  echo "$0 Starting (lda_mllt) triphone training in exp/tri2b"
  steps/train_lda_mllt.sh \
    --splice-opts "--left-context=3 --right-context=3" \
    2000 30000 \
    data/train data/lang exp/tri1_ali exp/tri2b

  (
  #  make decoding FSTs for tri2b models
  utils/mkgraph.sh \
    data/lang_test exp/tri2b exp/tri2b/graph || exit 1;

  # decode  test with tri2b models
  for x in native nonnative devtest test; do
    steps/decode.sh \
      --nj 8  \
      exp/tri2b/graph data/$x exp/tri2b/decode_${x} || exit 1;
  done
  ) &

  # align with lda and mllt adapted triphones
  steps/align_si.sh \
    --use-graphs true --nj 8 --cmd "$train_cmd" \
    data/train data/lang exp/tri2b exp/tri2b_ali

  echo "$0 Starting (SAT) triphone training in exp/tri3b"
  steps/train_sat.sh \
    --cmd "$train_cmd" \
    3100 50000 \
    data/train data/lang exp/tri2b_ali exp/tri3b

  # align with tri3b models
  echo "$0 Starting exp/tri3b_ali"
  steps/align_fmllr.sh \
    --nj 8 --cmd "$train_cmd" \
    data/train data/lang exp/tri3b exp/tri3b_ali
fi

if [ $stage -le 8 ]; then
  (
  # make decoding graphs for SAT models
  utils/mkgraph.sh \
    data/lang_test exp/tri3b exp/tri3b/graph ||  exit 1;

  # decode test sets with tri3b models
  for x in native nonnative devtest test; do
    steps/decode_fmllr.sh \
      --nj 8 --cmd "$decode_cmd" \
      exp/tri3b/graph data/$x exp/tri3b/decode_${x}
  done
  ) &
fi

if [ $stage -le 9 ]; then
  # train and test chain models
  local/chain/run_tdnn.sh
fi

wait
