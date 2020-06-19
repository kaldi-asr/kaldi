#!/usr/bin/env bash

# Copyright 2017-2018  Johns Hopkins University (Jan "Yenda" Trmal<jtrmal@gmail.com>)
#           2017-2018  Johns Hopkins University (author: Daniel Povey)
#                2018  Yiming Wang
#                2019  Mahsa Yarmohammadi
# License: Apache 2.0

. ./path.sh
. ./cmd.sh

nj=30 # number of parallel jobs
stage=1
language=swahili
. utils/parse_options.sh

set -e -o pipefail
set -o nounset                              # Treat unset variables as an error

[ ! -f ./conf/lang/${language}.conf ] && \
  echo "Language configuration conf/lang/${language}.conf does not exist!" && exit 1
ln -sf ./conf/lang/${language}.conf lang.conf
. ./lang.conf

if [ $stage -le 1 ]; then
  local/prepare_text_data.sh $corpus
  local/prepare_audio_data.sh $corpus
fi

if [ $stage -le 2 ]; then
  local/prepare_dict.sh $corpus
  utils/validate_dict_dir.pl data/local/dict_nosp
  utils/prepare_lang.sh data/local/dict_nosp \
    "<unk>" data/local/lang_nosp data/lang_nosp
  utils/validate_lang.pl data/lang_nosp
fi

if [ $stage -le 3 ]; then
  local/train_lms_srilm.sh --oov-symbol "<unk>" --words-file \
    data/lang_nosp/words.txt data data/lm
  utils/format_lm.sh data/lang_nosp data/lm/lm.gz \
    data/local/dict_nosp/lexiconp.txt data/lang_nosp_test
  utils/validate_lang.pl data/lang_nosp_test
fi

if [ $stage -le 4 ]; then
  for set in train dev; do
    dir=data/$set
    utils/fix_data_dir.sh $dir
    steps/make_mfcc.sh --cmd "$train_cmd" --nj 16 $dir
    steps/compute_cmvn_stats.sh $dir
    utils/fix_data_dir.sh $dir
    utils/validate_data_dir.sh $dir
  done
fi

# Create a subset with 40k short segments to make flat-start training easier
if [ $stage -le 5 ]; then
  utils/subset_data_dir.sh --shortest data/train $numShorestUtts data/train_short
fi

# monophone training
if [ $stage -le 6 ]; then
  steps/train_mono.sh --nj $nj --cmd "$train_cmd" \
    data/train_short data/lang_nosp_test exp/mono
  (
    utils/mkgraph.sh data/lang_nosp_test \
      exp/mono exp/mono/graph_nosp
    for test in dev; do
      steps/decode.sh --nj $nj --cmd "$decode_cmd" exp/mono/graph_nosp \
        data/$test exp/mono/decode_nosp_$test
    done
  )&

  steps/align_si.sh --nj $nj --cmd "$train_cmd" \
    data/train data/lang_nosp_test exp/mono exp/mono_ali
fi

# train a first delta + delta-delta triphone system on all utterances
if [ $stage -le 7 ]; then
  steps/train_deltas.sh --cmd "$train_cmd" \
    $numLeavesTri1 $numGaussTri1 data/train data/lang_nosp_test exp/mono_ali exp/tri1

  # decode using the tri1 model
  (
    utils/mkgraph.sh data/lang_nosp_test exp/tri1 exp/tri1/graph_nosp
    for test in dev; do
      steps/decode.sh --nj $nj --cmd "$decode_cmd" exp/tri1/graph_nosp \
        data/$test exp/tri1/decode_nosp_$test
    done
  )&

  steps/align_si.sh --nj $nj --cmd "$train_cmd" \
    data/train data/lang_nosp_test exp/tri1 exp/tri1_ali
fi

# train an LDA+MLLT system.
if [ $stage -le 8 ]; then
  steps/train_lda_mllt.sh --cmd "$train_cmd" \
    --splice-opts "--left-context=3 --right-context=3" $numLeavesTri2 $numGaussTri2 \
    data/train data/lang_nosp_test exp/tri1_ali exp/tri2

  # decode using the LDA+MLLT model
  (
    utils/mkgraph.sh data/lang_nosp_test exp/tri2 exp/tri2/graph_nosp
    for test in dev; do
      steps/decode.sh --nj $nj --cmd "$decode_cmd" exp/tri2/graph_nosp \
        data/$test exp/tri2/decode_nosp_$test
    done
  )&

  steps/align_si.sh  --nj $nj --cmd "$train_cmd" --use-graphs true \
    data/train data/lang_nosp_test exp/tri2 exp/tri2_ali
fi

# Train tri3, which is LDA+MLLT+SAT
if [ $stage -le 9 ]; then
  steps/train_sat.sh --cmd "$train_cmd" $numLeavesTri3 $numGaussTri3 \
    data/train data/lang_nosp_test exp/tri2_ali exp/tri3

  # decode using the tri3 model
  (
    utils/mkgraph.sh data/lang_nosp_test exp/tri3 exp/tri3/graph_nosp
    for test in dev; do
      steps/decode_fmllr.sh --nj $nj --cmd "$decode_cmd" exp/tri3/graph_nosp \
        data/$test exp/tri3/decode_nosp_$test
    done
  )&
fi

# Now we compute the pronunciation and silence probabilities from training data,
# and re-create the lang directory.
if [ $stage -le 10 ]; then
  steps/get_prons.sh --cmd "$train_cmd" data/train data/lang_nosp_test exp/tri3
  utils/dict_dir_add_pronprobs.sh --max-normalize true \
    data/local/dict_nosp \
    exp/tri3/pron_counts_nowb.txt exp/tri3/sil_counts_nowb.txt \
    exp/tri3/pron_bigram_counts_nowb.txt data/local/dict

  utils/prepare_lang.sh data/local/dict "<unk>" data/local/lang data/lang

  utils/format_lm.sh data/lang data/lm/lm.gz \
    data/local/dict/lexiconp.txt data/lang_test

  steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" \
    data/train data/lang_test exp/tri3 exp/tri3_ali
fi

if [ $stage -le 11 ]; then
  # Test the tri3 system with the silprobs and pron-probs.

  # decode using the tri3 model
  utils/mkgraph.sh data/lang_test exp/tri3 exp/tri3/graph
  for test in dev; do
    steps/decode_fmllr.sh --nj $nj --cmd "$decode_cmd" \
      exp/tri3/graph data/$test exp/tri3/decode_$test
  done
fi

mkdir -p data/bitext
mkdir -p data/mono

srctext_bitext=data/bitext/text
srctext_mono=data/mono/text

if [ $stage -le 12 ]; then
  # Read the foreign part of the bitext as $srctext_bitext and preprocess the text
  if [ "$number_mapping" != "" ]; then
    echo Number mapping file Found. Converting numbers...
    cat $bitext | awk -F"\t" '{print $2;}' | local/normalize_numbers.py $number_mapping > $srctext_bitext
    if [[ $mono == *.gz ]]; then 
      gzip -cd $mono | local/normalize_numbers.py $number_mapping > $srctext_mono
    else
      cat $mono | local/normalize_numbers.py $number_mapping > $srctext_mono
    fi
    if [ "$mono2" != "" ]; then
      if [[ $mono2 == *.gz ]]; then 
        gzip -cd $mono2 | local/normalize_numbers.py $number_mapping >> $srctext_mono
      else
        cat $mono2 | local/normalize_numbers.py $number_mapping >> $srctext_mono
      fi
    fi
  else
    cat $bitext | awk -F"\t" '{print $2;}' > $srctext_bitext
    if [[ $mono == *.gz ]]; then
      gzip -cd $mono > $srctext_mono
    else
      cat $mono > $srctext_mono
    fi
    if [ "$mono2" != "" ]; then
      if [[ $mono2 == *.gz ]]; then 
        gzip -cd $mono2 >> $srctext_mono
      else
        cat $mono2 >> $srctext_mono
      fi
    fi
  fi

  local/preprocess_external_text.sh --language $language \
    --srctext-bitext ${srctext_bitext} ${srctext_bitext}.txt

  local/preprocess_external_text.sh --language $language \
    --srctext-bitext ${srctext_mono} ${srctext_mono}.txt

  # Combine two sources of text
  cat $bitext | awk '{print $1}' > ${srctext_bitext}.header
  paste ${srctext_bitext}.header ${srctext_bitext}.txt > ${srctext_bitext}.processed

  if [[ $mono == *.gz ]]; then
    gzip -cd $mono | awk '{printf("mono-%d\n",NR)}' > ${srctext_mono}.header
  else
    cat $mono | awk '{printf("mono-%d\n",NR)}' > ${srctext_mono}.header
  fi
  if [ "$mono2" != "" ]; then
    if [[ $mono2 == *.gz ]]; then 
      gzip -cd $mono2 | awk '{printf("mono-%d\n",NR)}' >> ${srctext_mono}.header
    else
      cat $mono2 | awk '{printf("mono-%d\n",NR)}' >> ${srctext_mono}.header
    fi
  fi
  paste ${srctext_mono}.header ${srctext_mono}.txt > ${srctext_mono}.processed
fi

# The next 3 stages are to train g2p from the existing lexicon,
# apply g2p to expand the lexicon using oov words from bitext data
# as in ${dict_root}_nosp.
g2p_workdir=data/local/g2p_phonetisarus
if [ $stage -le 13 ]; then
  echo 'Gathering missing words...'
  mkdir -p ${g2p_workdir}
  cat ${srctext_bitext}.txt ${srctext_mono}.txt | \
    local/count_oovs.pl data/local/dict_nosp/lexicon.txt | \
    awk '{for(i=4; i<NF; i++) printf "%s",$i OFS; if(NF) printf "%s",$NF; printf ORS}' | \
    perl -ape 's/\s/\n/g;' | \
    sort | uniq > ${g2p_workdir}/missing.txt
  cat ${g2p_workdir}/missing.txt | \
    grep "^[a-z]*$"  > ${g2p_workdir}/missing_onlywords.txt
fi

if [ $stage -le 14 ]; then
  local/g2p/train_g2p.sh --stage 0 --silence-phones \
    "data/local/dict/silence_phones.txt" data/local/dict_nosp exp/g2p || touch exp/g2p/.error
fi

dict_root=data/local/dict_combined
if [ $stage -le 15 ]; then
  if [ -f exp/g2p/.error ]; then
    rm exp/g2p/.error || true
    echo "Fail to train the G2P model." && exit 1;
  fi
  mkdir -p ${dict_root}_nosp
  rm ${dict_root}_nosp/lexiconp.txt 2>/dev/null || true
  cp data/local/dict_nosp/{phones,oov,nonsilence_phones,silence_phones,optional_silence}.txt ${dict_root}_nosp
  local/g2p/apply_g2p.sh --var-counts 1 exp/g2p/model.fst ${g2p_workdir} \
  data/local/dict_nosp/lexicon.txt ${dict_root}_nosp/lexicon.txt || exit 1;

  utils/validate_dict_dir.pl ${dict_root}_nosp
fi

lang_root=data/lang_combined
lmdir=data/lm_combined
if [ $stage -le 16 ]; then
  utils/prepare_lang.sh ${dict_root}_nosp "<unk>" data/local/lang_combined_nosp ${lang_root}_nosp
  utils/validate_lang.pl ${lang_root}_nosp
fi

# prepare the new LM with bitext data and the new lexicon,
# as in the new test lang directory ${lang_root}_nosp_test

datadev="data/analysis1 data/analysis2 data/test_dev data/eval1 data/eval2 data/eval3"

if [ $stage -le 17 ]; then
  for datadir in $datadev; do
    local/preprocess_test.sh $datadir &
  done
  wait

  mkdir -p $lmdir
  mkdir -p $lmdir/mono
  mkdir -p $lmdir/bitext

  cat data/analysis1/text | awk '{for(i=2;i<=NF;i++) printf("%s ", $i); print""}' \
    | grep . | shuf | head -n 2000 > $lmdir/dev_text || echo done

  local/train_lms_srilm.sh --oov-symbol "<unk>" --words-file ${lang_root}_nosp/words.txt \
    --train-text ${srctext_bitext}.processed --dev-text $lmdir/dev_text \
    data $lmdir/bitext

  local/train_lms_srilm.sh --oov-symbol "<unk>" --words-file ${lang_root}_nosp/words.txt \
    --train-text ${srctext_mono}.processed --dev-text $lmdir/dev_text \
    data $lmdir/mono
fi

if [ $stage -le 18 ]; then
  ngram -order 4 -lm data/lm/lm.gz -mix-lm $lmdir/bitext/lm.gz \
    -mix-lm2 $lmdir/mono/lm.gz -lambda 0.3 -mix-lambda2 0.4 \
    -write-lm $lmdir/lm.gz

  utils/format_lm.sh ${lang_root}_nosp $lmdir/lm.gz \
    ${dict_root}_nosp/lexiconp.txt ${lang_root}_nosp_test
  utils/validate_lang.pl ${lang_root}_nosp_test
fi

# Now we compute the pronunciation and silence probabilities from training data,
# and re-create the lang directory ${lang_root}_test.
if [ $stage -le 19 ]; then
  steps/get_prons.sh --cmd "$train_cmd" data/train ${lang_root}_nosp_test exp/tri3
  utils/dict_dir_add_pronprobs.sh --max-normalize true \
    ${dict_root}_nosp \
    exp/tri3/pron_counts_nowb.txt exp/tri3/sil_counts_nowb.txt \
    exp/tri3/pron_bigram_counts_nowb.txt ${dict_root}
  utils/prepare_lang.sh ${dict_root} "<unk>" data/local/lang_combined ${lang_root}

  utils/format_lm.sh ${lang_root} $lmdir/lm.gz \
    ${dict_root}/lexiconp.txt ${lang_root}_test
fi

# After run.sh is finished, run the followings:
# ./local/chain/run_tdnn.sh
# ./local/chain/decode_test.sh --language <swahili|tagalog|somali>
# ./local/rnnlm/run_tdnn_lstm.sh
exit 0;
