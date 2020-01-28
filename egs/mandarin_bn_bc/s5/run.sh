#!/usr/bin/env bash

# Copyright 2019 Johns Hopkins University (author: Jinyi Yang)
# Apache 2.0

train_nj=80
decode_nj=60
stage=-1

[ -f ./path.sh ] && . ./path.sh
[ -f ./cmd.sh ] && . ./cmd.sh
. parse_options.sh

GALE_AUDIO=(
  /export/corpora/LDC/LDC2013S08/
  /export/corpora/LDC/LDC2013S04/
  /export/corpora/LDC/LDC2014S09/
  /export/corpora/LDC/LDC2015S06/
  /export/corpora/LDC/LDC2015S13/
  /export/corpora/LDC/LDC2016S03/
  /export/corpora/LDC/LDC2017S25/
)
GALE_TEXT=(
  /export/corpora/LDC/LDC2013T20/
  /export/corpora/LDC/LDC2013T08/
  /export/corpora/LDC/LDC2014T28/
  /export/corpora/LDC/LDC2015T09/
  /export/corpora/LDC/LDC2015T25/
  /export/corpora/LDC/LDC2016T12/
  /export/corpora/LDC/LDC2017T18/
)

TDT_AUDIO=(
  /export/corpora/LDC/LDC2001S93/
  /export/corpora/LDC/LDC2001S95/
  /export/corpora/LDC/LDC2005S11/
)
TDT_TEXT=(
  /export/corpora/LDC/LDC2001T57/
  /export/corpora/LDC/LDC2001T58/
  /export/corpora5/LDC/LDC2005T16/
)

GIGA_TEXT=/export/corpora/LDC/LDC2003T09/gigaword_man/xin/

galeData=GALE/
tdtData=TDT/
gigaData=GIGA/

set -e -o pipefail
set -x

########################### Data preparation ###########################
if [ $stage -le 0 ]; then
  echo "`date -u`: Prepare data for GALE"
  local/gale_data_prep_audio.sh "${GALE_AUDIO[@]}" $galeData
  local/gale_data_prep_txt.sh  "${GALE_TEXT[@]}" $galeData
  local/gale_data_prep_split.sh $galeData data/local/gale

  echo "`date -u`: Prepare data for TDT"
  local/tdt_mandarin_data_prep_audio.sh "${TDT_AUDIO[@]}" $tdtData
  local/tdt_mandarin_data_prep_txt.sh  "${TDT_TEXT[@]}" $tdtData
  local/tdt_mandarin_data_prep_filter.sh $tdtData data/local/tdt_mandarin

  ## Merge transcripts from GALE and TDT for lexicon and LM training
  mkdir -p data/local/gale_tdt_train
  cat data/local/gale/train/text data/local/tdt_mandarin/text > data/local/gale_tdt_train/text
fi

########################### Lexicon preparation ########################
if [ $stage -le 1 ]; then
  echo "`date -u`: Prepare dictionary for GALE and TDT"
  local/mandarin_prepare_dict.sh data/local/dict_gale_tdt data/local/gale_tdt_train
  local/check_oov_rate.sh data/local/dict_gale_tdt/lexicon.txt \
    data/local/gale_tdt_train/text > data/local/gale_tdt_train/oov.rate
  grep "rate" data/local/gale_tdt_train/oov.rate |\
    awk '$10>0{print "Warning: OOV rate is "$10 ", make sure it is a small number"}'
  utils/prepare_lang.sh data/local/dict_gale_tdt "<UNK>" data/local/lang_gale_tdt data/lang_gale_tdt
fi

########################### LM preparation for GALE ####################
if [ $stage -le 2 ]; then
  echo "`date -u`: Creating LM for GALE"
  local/mandarin_prepare_lm.sh --no-uttid "false" --ngram-order 4 --oov-sym "<UNK>" --prune_thres "1e-9" \
    data/local/dict_gale_tdt data/local/gale/train data/local/gale/train/lm_4gram data/local/gale/dev
  local/mandarin_format_lms.sh data/local/gale/train/lm_4gram/srilm.o4g.kn.gz \
    data/lang_gale_tdt data/lang_gale_test
fi

############# Using GALE data to train cleaning up model for TDT #######
datadir=data/gale
mfccdir=mfcc/gale
expdir=exp/gale
if [ $stage -le 3 ]; then
  # spread the mfccs over various machines, as this data-set is quite large.
  if [[  $(hostname -f) ==  *.clsp.jhu.edu ]]; then
    mfcc=$(basename $mfccdir) # in case was absolute pathname (unlikely), get basename.
    utils/create_split_dir.pl /export/b{05,06,07,08}/$USER/kaldi-data/egs/gale_asr/s5/$mfcc/storage \
      $mfccdir/storage
  fi
  echo "`date -u`: Extracting GALE MFCC features"
  for x in train dev eval; do
    steps/make_mfcc_pitch.sh --cmd "$train_cmd" --nj $train_nj \
      $datadir/$x exp/make_mfcc/gale/$x $mfccdir
    utils/fix_data_dir.sh $datadir/$x # some files fail to get mfcc for many reasons
    steps/compute_cmvn_stats.sh $datadir/$x exp/make_mfcc/gale/$x $mfccdir
  done
# Let's create small subsets to make quick flat-start training:
# train_100k contains about 150 hours of data.
	utils/subset_data_dir.sh $datadir/train 100000 $datadir/train_100k || exit 1;
	utils/subset_data_dir.sh --shortest $datadir/train_100k 2000 $datadir/train_2k_short || exit 1;
	utils/subset_data_dir.sh $datadir/train_100k 5000 $datadir/train_5k || exit 1;
	utils/subset_data_dir.sh $datadir/train_100k 10000 $datadir/train_10k || exit 1;
fi

########################### Monophone training #########################
if [ $stage -le 4 ]; then
  echo "`date -u`: Monophone trainign with GALE data"
	steps/train_mono.sh --boost-silence 1.25 --nj $train_nj --cmd "$train_cmd" \
  $datadir/train_2k_short data/lang_gale_tdt $expdir/mono || exit 1;
fi

########################### Tri1 training ##############################
if [ $stage -le 5 ]; then
  steps/align_si.sh --boost-silence 1.25 --nj $train_nj --cmd "$train_cmd" \
    $datadir/train_5k data/lang_gale_tdt $expdir/mono $expdir/mono_ali_5k || exit 1;
  echo "`date -u`: Tri1 trainign with GALE data"
	# train tri1 [first triphone pass]
	steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd" \
  	2000 10000 $datadir/train_5k data/lang_gale_tdt $expdir/mono_ali_5k $expdir/tri1 || exit 1;
	utils/mkgraph.sh data/lang_gale_test $expdir/tri1 $expdir/tri1/graph_gale_test || exit 1;
	steps/decode.sh  --nj $decode_nj --cmd "$decode_cmd" \
  	$expdir/tri1/graph_gale_test $datadir/dev $expdir/tri1/decode_gale_dev
fi

########################### Tri2b training #############################
if [ $stage -le 6 ]; then
	steps/align_si.sh --nj $train_nj --cmd "$train_cmd" \
  	$datadir/train_10k data/lang_gale_tdt $expdir/tri1 $expdir/tri1_ali_10k || exit 1;
  echo "`date -u`: Tri2b trainign with GALE data"
	steps/train_lda_mllt.sh --cmd "$train_cmd" \
                          --splice-opts "--left-context=3 --right-context=3" 2500 15000 \
                          $datadir/train_10k data/lang_gale_tdt $expdir/tri1_ali_10k $expdir/tri2b
	utils/mkgraph.sh data/lang_gale_test $expdir/tri2b $expdir/tri2b/graph_gale_test || exit 1;
	steps/decode.sh  --nj $decode_nj --cmd "$decode_cmd" \
  	$expdir/tri2b/graph_gale_test $datadir/dev $expdir/tri2b/decode_gale_dev
fi

########################### Tri3b training #############################
if [ $stage -le 7 ]; then
	steps/align_si.sh --nj $train_nj --cmd "$train_cmd" --use-graphs true \
  	$datadir/train_10k data/lang_gale_tdt $expdir/tri2b $expdir/tri2b_ali_10k || exit 1;
  echo "`date -u`: Tri3b trainign with GALE data"
	steps/train_sat.sh --cmd "$train_cmd" 2500 15000 \
                     $datadir/train_10k data/lang_gale_tdt $expdir/tri2b_ali_10k $expdir/tri3b
	utils/mkgraph.sh data/lang_gale_test $expdir/tri3b $expdir/tri3b/graph_gale_test || exit 1;
	steps/decodei_fmllr.sh  --nj $decode_nj --cmd "$decode_cmd" \
  	$expdir/tri3b/graph_gale_test $datadir/dev $expdir/tri3b/decode_gale_dev
fi

########################### Tri4b training #############################
if [ $stage -le 8 ]; then
	steps/align_fmllr.sh --nj $train_nj --cmd "$train_cmd" \
    $datadir/train_100k data/lang_gale_tdt \
    $expdir/tri3b $expdir/tri3b_ali_100k || exit 1;
  echo "`date -u`: Tri4b trainign with GALE data"
	steps/train_sat.sh  --cmd "$train_cmd" 4200 40000 \
                      $datadir/train_100k data/lang_gale_tdt \
                      $expdir/tri3b_ali_100k $expdir/tri4b
	utils/mkgraph.sh data/lang_gale_test $expdir/tri4b $expdir/tri4b/graph_gale_test || exit 1;
  steps/decode_fmllr.sh  --nj $decode_nj --cmd "$decode_cmd" \
  	$expdir/tri4b/graph_gale_test $datadir/dev $expdir/tri4b/decode_gale_dev
fi

######################### Re-create lang directory######################
# We want to add pronunciation probabilities to lexicon, using the  previously trained model.
if [ $stage -le 9 ]; then
	steps/get_prons.sh --cmd "$train_cmd" \
                     $datadir/train_100k data/lang_gale_tdt $expdir/tri4b
  utils/dict_dir_add_pronprobs.sh --max-normalize true \
                                  data/local/dict_gale_tdt \
                                  $expdir/tri4b/pron_counts_nowb.txt $expdir/tri4b/sil_counts_nowb.txt \
                                  $expdir/tri4b/pron_bigram_counts_nowb.txt data/local/dict_gale_tdt_reestimated
  utils/prepare_lang.sh data/local/dict_gale_tdt_reestimated \
                        "<UNK>" data/local/lang_gale_tdt_reestimated data/lang_gale_tdt_reestimated
  local/mandarin_format_lms.sh data/local/gale/train/lm_4gram/srilm.o4g.kn.gz \
    data/lang_gale_tdt_reestimated data/lang_gale_tdt_reestimated_test
fi

######################### Train tri5b with all GALE data ###############
if [ $stage -le 10 ]; then
	steps/align_fmllr.sh --nj $train_nj --cmd "$train_cmd" \
    $datadir/train data/lang_gale_tdt_reestimated \
    $expdir/tri4b $expdir/tri4b_ali_train || exit 1;

	steps/train_sat.sh  --cmd "$train_cmd" 5000 100000 \
    $datadir/train data/lang_gale_tdt_reestimated \
		$expdir/tri4b_ali_train $expdir/tri5b || exit 1;
fi

if [ $stage -le 11 ]; then
  echo "Clean up TDT data"
  mkdir -p data/tdt || exit 1;
  mfccdir=mfcc/tdt
  cp -r data/local/tdt_mandarin/* data/tdt
  steps/make_mfcc_pitch.sh --cmd "$train_cmd" --nj $train_nj \
      data/tdt exp/make_mfcc/tdt $mfccdir
  utils/fix_data_dir.sh data/tdt # some files fail to get mfcc for many reasons
  steps/compute_cmvn_stats.sh data/tdt exp/make_mfcc/tdt $mfccdir
  local/tdt_cleanup.sh --nj $train_nj data/tdt data/lang_gale_tdt_reestimated \
    $expdir/tri5b $expdir/tri5b_tdt_cleanup data/tdt_cleanup
  sed -i 's/<UNK>//g' data/tdt_cleanup/text
  steps/compute_cmvn_stats.sh data/tdt_cleanup exp/make_mfcc/tdt_cleanup ${mfccdir}_cleanup
fi

datadir=data/train_gale_tdt_cleanup
expdir=exp
if [ $stage -le 12 ]; then
  echo "Combine GALE and TDT cleaned"
	utils/combine_data.sh \
    $datadir data/gale/train data/tdt_cleanup

	steps/align_fmllr.sh --nj $train_nj --cmd "$train_cmd" \
    $datadir data/lang_gale_tdt_reestimated \
    exp/gale/tri5b exp/gale/tri5b_ali_gale_tdt_cleanup || exit 1;

	steps/train_quick.sh --cmd "$train_cmd" \
    7000 150000 $datadir data/lang_gale_tdt_reestimated \
		exp/gale/tri5b_ali_gale_tdt_cleanup exp/tri6b_cleanup
  utils/mkgraph.sh data/lang_gale_tdt_reestimated_test exp/tri6b_cleanup \
    exp/tri6b_cleanup/graph_gale_tdt_reestimated_test || exit 1;
  steps/decode_fmllr.sh  --nj $decode_nj --cmd "$decode_cmd" \
    exp/tri6b_cleanup/graph_gale_tdt_reestimated_test data/gale/dev exp/tri6b_cleanup/decode_gale_dev
fi

if [ $stage -le 13 ]; then
  echo "Expand the lexicon with Gigaword"
  local/gigaword_prepare.sh $GIGA_TEXT $gigaData
  local/mandarin_prepare_dict.sh data/local/dict_giga_man_simp data/local/giga_man_simp
  utils/prepare_lang.sh data/local/dict_giga_man_simp "<UNK>" \
    data/local/lang_giga_man_simp data/lang_giga_man_simp
  # Merge the previous dictionary with GIGAWORD dictionary
  local/mandarin_merge_dict.sh data/local/dict_gale_tdt_reestimated data/local/dict_giga_man_simp data/local/dict_large
  # Prune the lexicon for multi-pronunciation words
  python3 local/prune_lex.py data/local/dict_large/lexiconp.txt | \
    sort > data/local/dict_large/lexiconp.tmp
  mv data/local/dict_large/lexiconp.tmp data/local/dict_large/lexiconp.txt
  utils/prepare_lang.sh data/local/dict_large "<UNK>" \
    data/local/lang_large data/lang_large
fi


if [ $stage -le 14 ]; then
  echo "Prepare LM with all data"
  # Train LM with GALE + TDT
  local/mandarin_prepare_lm.sh --no-uttid "false" --ngram-order 4 --oov-sym "<UNK>" --prune_thres "1e-9" \
    data/local/dict_large data/local/gale_tdt_train data/local/gale_tdt_lm_4gram data/local/gale/dev

  # Train LM with gigaword
  local/mandarin_prepare_lm.sh --no-uttid "true" --ngram-order 4 --oov-sym "<UNK>" --prune_thres "1e-9" \
    data/local/dict_large GIGA/ data/local/giga_lm_4gram data/local/gale/dev

  # LM interpolation
  local/mandarin_mix_lm.sh --ngram-order 4 --oov-sym "<UNK>" --prune-thres "1e-9" \
    data/local/gale_tdt_lm_4gram data/local/giga_lm_4gram data/local/lm_large_4gram data/local/gale/dev
  local/mandarin_format_lms.sh data/local/lm_large_4gram/srilm.o4g.kn.gz \
    data/lang_large data/lang_large_test
fi

# From here, we train a tdnnf model. You should modify the related directories
# in this script, and in local/nnet3/run_ivector_common.sh
local/chain/run_tdnn.sh

# We use all GALE+TDT+GIGAWORD text to train RNNLM
cat local/gale_tdt_lm_4gram/text data/local/giga_lm_4gram/text | gzip > data/local/lm_large_4gram/train_text.gz
# Train RNNLM. You should modify the related directories in this script.
local/rnnlm/run_tdnn_lstm_1a.sh

