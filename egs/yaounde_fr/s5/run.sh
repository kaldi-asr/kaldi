#!/bin/bash 

# Uses the cmusphinx French lexicon.

. ./cmd.sh
. ./path.sh
stage=0

. ./utils/parse_options.sh

set -e
set -o pipefail
set u

# Do not change tmpdir, other scripts under local depend on it
tmpdir=data/local/tmp

# location of corpora
# The speech corpus is on openslr.org
speech="http://www.openslr.org/resources/57/African_Accented_French.tar.gz"

datadir=African_Accented_French

# We use the cmusphinx lexicon.
lex='https://sourceforge.net/projects/cmusphinx/files/Acoustic and Language Models/French/fr.dict/download'

# We train a large lm on subtitles.
subs_src="http://opus.nlpl.eu/download.php?f=OpenSubtitles2018/mono/OpenSubtitles2018.fr.gz"

if [ $stage -le 1 ]; then
  # Downloads archive to this script's directory
  local/aafr_download.sh $speech

  local/cmusphinx_fr_lexicon_download.sh $lex

  local/subs/download.sh $subs_src
fi

# preparation stages will store files under data/
# Delete the entire data directory when restarting.
if [ $stage -le 2 ]; then
  local/prepare_data.sh $datadir
fi

if [ $stage -le 3 ]; then
  echo "$0: Preparing initial dictionary."
  mkdir -p $tmpdir/dict
  local/prepare_dict.sh ./fr.dict data/local/dict_nosp
fi

if [ $stage -le 4 ]; then
  echo "$0: Training g2p model."
  local/g2p/train_g2p.sh data/local/dict_nosp $tmpdir/g2p
fi

if [ $stage -le 5 ]; then
  local/g2p/apply_g2p.sh $tmpdir/g2p/model.fst $tmpdir/dict data/local/dict_nosp/lexicon.txt $tmpdir/dict/lexicon.txt
fi

if [ $stage -le 6 ]; then
  local/prepare_dict.sh $tmpdir/dict/lexicon.txt data/local/dict_nosp_expanded
  echo "<UNK> SPN" >> data/local/dict_nosp/lexicon.txt
  echo "<UNK> SPN" >> data/local/dict_nosp_expanded/lexicon.txt
fi

if [ $stage -le 7 ]; then
  # prepare the lang directory
  utils/prepare_lang.sh data/local/dict_nosp_expanded "<UNK>" \
  data/local/lang_tmp_nosp_expanded data/lang_nosp_expanded
fi

if [ $stage -le 8 ]; then
  echo "Small lm training."
  mkdir -p $tmpdir/lm
  cut -d " " -f 2- data/train/text > $tmpdir/lm/train.txt
  local/prepare_small_lm.sh  $tmpdir/lm/train.txt
  echo "Making small G.fst."
  mkdir -p data/lang_nosp_expanded_test_tgsmall
  utils/format_lm.sh data/lang_nosp_expanded data/local/lm/tgsmall.arpa.gz \
    data/local/dict_nosp_expanded/lexicon.txt data/lang_nosp_expanded_test_tgsmall
fi

if [ $stage -le 9 ]; then
  echo "Preparing the subs data for larger lm training."
  # Subs prep depends on previous steps. 
  local/subs/prepare_data.pl 
  local/prepare_medium_lm.sh  $tmpdir/subs/lm/in_vocabulary.txt
  local/prepare_large_lm.sh  $tmpdir/subs/lm/in_vocabulary.txt
fi

if [ $stage -le 10 ]; then
  echo "Prepare medium size lang directory."
  mkdir -p data/lang_nosp_expanded_test_tgmed
  utils/format_lm.sh data/lang_nosp_expanded data/local/lm/tgmed.arpa.gz \
    data/local/dict_nosp_expanded/lexicon.txt data/lang_nosp_expanded_test_tgmed
  # Create ConstArpaLm format language model for full 3-gram and 4-gram LMs
  utils/build_const_arpa_lm.sh data/local/lm/tglarge.arpa.gz \
    data/lang_nosp_expanded data/lang_nosp_expanded_test_tglarge
fi

if [ $stage -le 11 ]; then
  # extract acoustic features
  for f in devtest dev train test unsup; do
    steps/make_mfcc.sh --cmd "$train_cmd" --nj 9 data/$f exp/make_mfcc/$f mfcc
    utils/fix_data_dir.sh data/$f
    steps/compute_cmvn_stats.sh data/$f exp/make_mfcc mfcc
    utils/fix_data_dir.sh data/$f
  done
fi

if [ $stage -le 12 ]; then
  # Get the shortest 500 utterances first because those are more likely
  # to have accurate alignments.
  utils/subset_data_dir.sh --shortest data/train 500 data/train_500short

  echo "$0: monophone training"
  steps/train_mono.sh  --cmd "$train_cmd" --nj 10 data/train_500short \
    data/lang_nosp_expanded exp/mono
fi

if [ $stage -le 13 ]; then
  echo "monophone evaluation"
  (
    # make decoding graph for monophones
    utils/mkgraph.sh data/lang_nosp_expanded_test_tgsmall exp/mono \
      exp/mono/graph_nosp_expanded_tgsmall

    # test monophones
    for x in devtest dev test unsup; do
      nspk=$(wc -l < data/$x/spk2utt)
      steps/decode.sh  --cmd "$decode_cmd" --nj $nspk \
        exp/mono/graph_nosp_expanded_tgsmall data/$x exp/mono/decode_nosp_expanded_tgsmall_${x}
    done
  ) &
fi

if [ $stage -le 14 ]; then
      echo "monophone evaluation with tgmed"
  (
    # make decoding graph for monophones
    utils/mkgraph.sh data/lang_nosp_expanded_test_tgmed exp/mono \
      exp/mono/graph_nosp_expanded_tgmed

    # test monophones
    for x in devtest dev test unsup; do
      nspk=$(wc -l < data/$x/spk2utt)
      steps/decode.sh  --cmd "$decode_cmd" --nj $nspk \
        exp/mono/graph_nosp_expanded_tgmed data/$x exp/mono/decode_nosp_expanded_tgmed_${x}
    done
  ) &
fi

if [ $stage -le 15 ]; then
  echo "$0: aligning with monophones"
  steps/align_si.sh  --cmd "$train_cmd" --nj 10 data/train data/lang_nosp_expanded \
    exp/mono exp/mono_ali
fi

if [ $stage -le 16 ]; then
  echo "$0: Starting  triphone training in exp/tri1"
  steps/train_deltas.sh \
    --cmd "$train_cmd" \
    --boost-silence 1.25 \
    3000 12000 \
    data/train data/lang_nosp_expanded exp/mono_ali exp/tri1
fi

wait

if [ $stage -le 17 ]; then
  echo "$0: testing cd gmm hmm models"
  (
    # make decoding graphs for tri1
    utils/mkgraph.sh data/lang_nosp_expanded_test_tgsmall exp/tri1 \
      exp/tri1/graph_nosp_expanded_tgsmall

    echo "Decoding test data with tri1 an tgsmall dmodels."
    for x in devtest dev test unsup; do
      nspk=$(wc -l < data/$x/spk2utt)
      steps/decode.sh --cmd "$decode_cmd" --nj $nspk exp/tri1/graph_nosp_expanded_tgsmall \
        data/$x exp/tri1/decode_nosp_expanded_tgsmall_${x}
      echo "$0: testing with cd gmm hmm tgmed and tglarge models"
      # make decoding graphs for tri1
      utils/mkgraph.sh data/lang_nosp_expanded_test_tgmed exp/tri1 \
        exp/tri1/graph_nosp_expanded_tgmed

      echo "Decoding test data with tri1 tgmed an tglarge dmodels."
      nspk=$(wc -l < data/$x/spk2utt)
      steps/decode.sh --cmd "$decode_cmd" --nj $nspk exp/tri1/graph_nosp_expanded_tgmed \
        data/$x exp/tri1/decode_nosp_expanded_tgmed_${x}

      steps/lmrescore.sh --cmd "$decode_cmd" data/lang_nosp_expanded_test_{tgsmall,tgmed} \
        data/$x exp/tri1/decode_nosp_expanded_{tgsmall,tgmed}_$x

      steps/lmrescore_const_arpa.sh \
        --cmd "$decode_cmd" data/lang_nosp_expanded_test_{tgsmall,tglarge} \
        data/$x exp/tri1/decode_nosp_expanded_{tgsmall,tglarge}_$x
    done
  ) &
fi

if [ $stage -le 18 ]; then
  # align with triphones
  steps/align_si.sh  --cmd "$train_cmd" --nj 10 data/train data/lang_nosp_expanded \
    exp/tri1 exp/tri1_ali
fi

if [ $stage -le 19 ]; then
  echo "$0: Starting (lda_mllt) triphone training in exp/tri2b"
  steps/train_lda_mllt.sh \
    --cmd "$train_cmd" --splice-opts "--left-context=3 --right-context=3" \
    3500 16000 \
    data/train data/lang_nosp_expanded exp/tri1_ali exp/tri2b
fi

wait

if [ $stage -le 20 ]; then
  (
    echo "$0: Making decoding FSTs for tri2b models."
    utils/mkgraph.sh data/lang_nosp_expanded_test_tgsmall exp/tri2b \
      exp/tri2b/graph_nosp_expanded_tgsmall
    # decode  test with tri2b models
    for x in devtest dev dev test unsup; do
      nspk=$(wc -l < data/$x/spk2utt)
      steps/decode.sh --cmd "$decode_cmd" --nj $nspk \
        exp/tri2b/graph_nosp_expanded_tgsmall data/$x exp/tri2b/decode_nosp_expanded_tgsmall_${x}

      steps/lmrescore.sh --cmd "$decode_cmd" \
        data/lang_nosp_expanded_test_{tgsmall,tgmed} data/$x \
        exp/tri2b/decode_nosp_expanded_{tgsmall,tgmed}_$x

      steps/lmrescore_const_arpa.sh \
        --cmd "$decode_cmd" data/lang_nosp_expanded_test_{tgsmall,tglarge} \
        data/$x exp/tri2b/decode_nosp_expanded_{tgsmall,tglarge}_$x
    done
  )&
fi

if [ $stage -le 21 ]; then
  echo "$0: aligning with lda and mllt adapted triphones"
  steps/align_si.sh  --nj 10 \
    --cmd "$train_cmd" \
    --use-graphs true data/train data/lang_nosp_expanded exp/tri2b exp/tri2b_ali
fi

if [ $stage -le 22 ]; then
  echo "$0: Starting (SAT) triphone training in exp/tri3b"
  steps/train_sat.sh --cmd "$train_cmd" \
    4000 20000 \
    data/train data/lang_nosp_expanded exp/tri2b_ali exp/tri3b
fi

wait

if [ $stage -le 23 ]; then
  (
    echo "$0: making decoding graph for SAT models."
    utils/mkgraph.sh data/lang_nosp_expanded_test_tgsmall exp/tri3b \
      exp/tri3b/graph_nosp_expanded_tgsmall
    for x in devtest dev test unsup; do
      echo "Decoding $x with sat and tgsmall models."
      nspk=$(wc -l < data/$x/spk2utt)
      steps/decode_fmllr.sh --cmd "$decode_cmd" --nj $nspk \
        exp/tri3b/graph_nosp_expanded_tgsmall data/$x \
        exp/tri3b/decode_nosp_expanded_tgsmall_${x}

      steps/lmrescore.sh --cmd "$decode_cmd" \
        data/lang_nosp_expanded_test_{tgsmall,tgmed} data/$x \
        exp/tri3b/decode_nosp_expanded_{tgsmall,tgmed}_$x
      steps/lmrescore_const_arpa.sh \
        --cmd "$decode_cmd" data/lang_nosp_expanded_test_{tgsmall,tglarge} \
        data/$x exp/tri3b/decode_nosp_{expanded_tgsmall,tglarge}_$x
    done
  )&
fi

if [ $stage -le 24 ]; then
  echo "$0: Starting exp/tri3b_ali"
  steps/align_fmllr.sh --cmd "$train_cmd" --nj 10 data/train data/lang_nosp_expanded \
    exp/tri3b exp/tri3b_ali
fi

if [ $stage -le 25 ]; then
  #  compute the pronunciation and silence probabilities from training data,
  #and re-create the lang directory.
  steps/get_prons.sh --cmd "$train_cmd" \
    data/train data/lang_nosp_expanded exp/tri3b
fi

if [ $stage -le 26 ]; then
  utils/dict_dir_add_pronprobs.sh --max-normalize true \
    data/local/dict_nosp_expanded \
    exp/tri3b/pron_counts_nowb.txt exp/tri3b/sil_counts_nowb.txt \
    exp/tri3b/pron_bigram_counts_nowb.txt data/local/dict
fi

if [ $stage -le 27 ]; then
  utils/prepare_lang.sh data/local/dict \
    "<UNK>" data/local/lang_tmp data/lang
fi

if [ $stage -le 28 ]; then
  utils/format_lm.sh data/lang data/local/lm/tgsmall.arpa.gz \
    data/local/dict/lexicon.txt data/lang_test_tgsmall
fi

if [ $stage -le 29 ]; then
  utils/format_lm.sh data/lang data/local/lm/tgmed.arpa.gz \
    data/local/dict/lexicon.txt data/lang_test_tgmed
fi

if [ $stage -le 30 ]; then
  utils/build_const_arpa_lm.sh \
    data/local/lm/tglarge.arpa.gz data/lang data/lang_test_tglarge
fi

if [ $stage -le 31 ]; then
  echo "$0: Testing the tri3b system with the silprobs and pron-probs."
  # decode using the tri3b and tgsmall model
  (
    utils/mkgraph.sh data/lang_test_tgsmall \
      exp/tri3b exp/tri3b/graph_tgsmall
    for x in dev devtest test unsup; do
      steps/decode_fmllr.sh --nj 10 --cmd "$decode_cmd" \
        exp/tri3b/graph_tgsmall data/$x \
        exp/tri3b/decode_tgsmall_$x

      steps/lmrescore.sh --cmd "$decode_cmd" data/lang_test_{tgsmall,tgmed} \
        data/$x exp/tri3b/decode_{tgsmall,tgmed}_$x
      steps/lmrescore_const_arpa.sh \
        --cmd "$decode_cmd" data/lang_test_{tgsmall,tglarge} \
        data/$x exp/tri3b/decode_{tgsmall,tglarge}_$x
    done
  )&
fi

if [ $stage -le 32 ]; then
  # train and test chain models
  local/chain/run_tdnn.sh
fi

if [ $stage -le 33 ]; then
  # Run grammar decoding demos
  local/grammar/simple_demo.sh

  local/grammar/extend_vocab_demo.sh
fi
