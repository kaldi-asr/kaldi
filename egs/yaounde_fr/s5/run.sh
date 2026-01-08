#!/bin/bash 

. ./cmd.sh
. ./path.sh
stage=0

. ./utils/parse_options.sh

# Do not change tmpdir, other scripts under local depend on it
tmpdir=data/local/tmp

# location of corpora
# The speech corpus is on openslr.org
speech="http://www.openslr.org/resources/57/African_Accented_French.tar.gz"

datadir=African_Accented_French

# We use the cmusphinx lexicon.
lex='https://sourceforge.net/projects/cmusphinx/files/Acoustic and Language Models/French/fr.dict/download'

if [ $stage -le 1 ]; then
  echo "$0: Downloading archive to $(pwd)."
  local/aafr_download.sh $speech || exit 1;

  local/cmusphinx_fr_lexicon_download.sh $lex || exit 1;
fi

# preparation stages will store files under data/
# Delete the entire data directory when restarting.
if [ $stage -le 2 ]; then
  local/prepare_data.sh $datadir || exit 1;
fi

if [ $stage -le 3 ]; then
  echo "$0: Preparing initial dictionary."
  local/prepare_dict.sh ./fr.dict $tmpdir/dict_init || exit 1;
fi

if [ $stage -le 4 ]; then
  echo "$0: Training g2p model."
  local/g2p/train_g2p.sh $tmpdir/dict_init $tmpdir/g2p || exit 1;
fi

if [ $stage -le 5 ]; then
    local/g2p/apply_g2p.sh $tmpdir/g2p/model.fst $tmpdir/dict_work \
      $tmpdir/dict_init/lexicon.txt $tmpdir/dict_init/lexicon_with_tabs.txt \
      || exit 1;
  mkdir -p $tmpdir/dict
  expand -t 1 $tmpdir/dict_init/lexicon_with_tabs.txt > $tmpdir/dict/lexicon.txt
fi

if [ $stage -le 6 ]; then
  echo "$0: Preparing expanded lexicon."
  local/prepare_dict.sh $tmpdir/dict/lexicon.txt data/local/dict || exit 1;
  echo "<UNK> SPN" >> data/local/dict/lexicon.txt
fi

if [ $stage -le 7 ]; then
  echo "$0: Preparing the lang directory."
  utils/prepare_lang.sh data/local/dict "<UNK>" \
    data/local/lang data/lang || exit 1;
fi

if [ $stage -le 8 ]; then
  echo "$0: lm training."
  mkdir -p $tmpdir/lm
  cut -d " " -f 2- data/train/text > $tmpdir/lm/train.txt
  local/prepare_lm.sh  $tmpdir/lm/train.txt
  echo "$0: Making G.fst."
  utils/format_lm.sh data/lang data/local/lm/tg.arpa.gz \
    data/local/dict/lexicon.txt data/lang_test || exit 1;
fi

if [ $stage -le 9 ]; then
  for f in devtest dev train test; do
    echo "Extracting acoustic features for $f."
    steps/make_mfcc.sh --cmd "$train_cmd" --nj 4 data/$f exp/make_mfcc/$f mfcc
    utils/fix_data_dir.sh data/$f
    steps/compute_cmvn_stats.sh data/$f exp/make_mfcc mfcc
    utils/fix_data_dir.sh data/$f
  done
fi

if [ $stage -le 10 ]; then
  echo "$0: monophone training."
  steps/train_mono.sh  --cmd "$train_cmd" --nj 4 data/train data/lang exp/mono \
    || exit 1;
fi

if [ $stage -le 11 ]; then
  echo "$0: monophone evaluation"
  echo "$0: making decoding graph for monophones."
  utils/mkgraph.sh data/lang_test exp/mono exp/mono/graph || exit 1;
fi

if [ $stage -le 12 ]; then
  (
    for x in devtest test; do
      echo "Testing monophones  on $x."
      nspk=$(wc -l < data/$x/spk2utt)
      steps/decode.sh  --cmd "$decode_cmd" --nj $nspk exp/mono/graph data/$x \
        exp/mono/decode_${x} || exit 1;
    done
  ) &
fi

if [ $stage -le 13 ]; then
  echo "$0: aligning with monophones"
  steps/align_si.sh  --cmd "$train_cmd" --nj 4 data/train data/lang \
    exp/mono exp/mono_ali
fi

if [ $stage -le 14 ]; then
  echo "$0: Starting  triphone training in exp/tri1"
  steps/train_deltas.sh --cmd "$train_cmd" --boost-silence 1.25 \
    700 7720 \
    data/train data/lang exp/mono_ali exp/tri1 || exit 1;
fi

if [ $stage -le 13 ]; then
  echo "$0: Aligning with triphones."
  steps/align_si.sh  --cmd "$train_cmd" --nj 4 data/train data/lang \
    exp/tri1 exp/tri1_ali || exit 1;
fi

wait

if [ $stage -le 14 ]; then
  echo "$0: testing cd gmm hmm models"
  (
    echo "$0: Making decoding graphs for tri1."
    utils/mkgraph.sh data/lang_test exp/tri1 exp/tri1/graph || exit 1;

    for x in devtest test; do
      echo "$0: Decoding test data with tri1 on $x."
      nspk=$(wc -l < data/$x/spk2utt)
      steps/decode.sh --cmd "$decode_cmd" --nj $nspk \
        exp/tri1/graph data/$x \
        exp/tri1/decode_${x} || exit 1;
    done
  ) &
fi

if [ $stage -le 15 ]; then
  echo "$0: Starting lda_mllt triphone training in exp/tri2b."
  steps/train_lda_mllt.sh --cmd "$train_cmd" \
    --splice-opts "--left-context=3 --right-context=3" \
    700 7720 \
    data/train data/lang exp/tri1_ali exp/tri2b
fi

if [ $stage -le 16 ]; then
  echo "$0: aligning with lda and mllt adapted triphones"
  steps/align_si.sh  --nj 4 \
    --cmd "$train_cmd" \
    --use-graphs true data/train data/lang exp/tri2b exp/tri2b_ali
fi

wait

if [ $stage -le 17 ]; then
  (
    echo "$0: Making decoding FSTs for tri2b models."
    utils/mkgraph.sh data/lang_test exp/tri2b exp/tri2b/graph || exit 1;

    for x in devtest test; do
      nspk=$(wc -l < data/$x/spk2utt)
      steps/decode.sh --cmd "$decode_cmd" --nj $nspk \
        exp/tri2b/graph data/$x exp/tri2b/decode_${x} || exit 1;
    done
  ) &
fi

if [ $stage -le 18 ]; then
  echo "$0: Starting SAT triphone training in exp/tri3b."
  steps/train_sat.sh --cmd "$train_cmd" \
    700 7775 \
    data/train data/lang exp/tri2b_ali exp/tri3b
fi

wait

if [ $stage -le 19 ]; then
  echo "$0: making decoding graph for SAT models."
  (
    utils/mkgraph.sh data/lang_test exp/tri3b exp/tri3b/graph || exit 1;
    for x in devtest test; do
      echo "$0: Decoding $x with sat models."
      nspk=$(wc -l < data/$x/spk2utt)
      steps/decode_fmllr.sh --cmd "$decode_cmd" --nj $nspk \
        exp/tri3b/graph data/$x \
        exp/tri3b/decode_${x} || exit 1;
    done
  ) &
fi

if [ $stage -le 20 ]; then
  echo "$0: Starting exp/tri3b_ali."
  steps/align_fmllr.sh --cmd "$train_cmd" --nj 4 data/train data/lang \
    exp/tri3b exp/tri3b_ali || exit 1;
fi

wait

if [ $stage -le 21 ]; then
  echo "$0: Train and test chain models."
  local/chain/run_tdnn.sh
fi
