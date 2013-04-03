#!/bin/bash

# This is not necessarily the top-level run.sh as it is in other directories.   see README.txt first.


. conf/common_vars.sh || exit 1;
. ./lang.conf || exit 1;

[ -f local.conf ] && . ./local.conf

#Preparing dev2h and train directories
if [ ! -d data/raw_train_data ]; then
    echo ---------------------------------------------------------------------
    echo "Subsetting the TRAIN set"
    echo ---------------------------------------------------------------------

    local/make_corpus_subset.sh "$train_data_dir" "$train_data_list" ./data/raw_train_data || exit 1
    train_data_dir=`readlink -f ./data/raw_train_data`
fi

nj_max=`cat $train_data_list | wc -l`
if [[ "$nj_max" -lt "$train_nj" ]] ; then
  echo "The maximum reasonable number of jobs is $nj_max (you have $train_nj)! (The training and decoding process has file-granularity)"
  exit 1;
  train_nj=$nj_max
fi

if [ ! -d data/raw_dev2h_data ]; then
  echo ---------------------------------------------------------------------
  echo "Subsetting the DEV2H set"
  echo ---------------------------------------------------------------------
  
  local/make_corpus_subset.sh "$dev2h_data_dir" "$dev2h_data_list" ./data/raw_dev2h_data || exit 1
fi
dev2h_data_dir=`readlink -f ./data/raw_dev2h_data`

nj_max=`cat $dev2h_data_list | wc -l`
if [[ "$nj_max" -lt "$decode_nj" ]] ; then
  echo "The maximum reasonable number of jobs is $nj_max -- you have $decode_nj! (The training and decoding process has file-granularity)"
  exit 1
  decode_nj=$nj_max
fi

#if [[ $filter_lexicon ]]; then
#    lexicon_dir=./data/raw_lex_data
#    mkdir -p $lexicon_dir
#    if [[ ! -f $lexicon_dir/lexicon.txt ||  $lexicon_dir/lexicon.txt -ot $train_data_dir/transcription ]]; then 
#      echo ---------------------------------------------------------------------
#      echo "Subsetting the LEXICON"
#      echo ---------------------------------------------------------------------
#      local/make_lexicon_subset.sh $train_data_dir/transcription \
#        $lexicon_file $lexicon_dir/lexicon.txt || exit 1
#    fi
#    lexicon_file=$lexicon_dir/lexicon.txt
#fi


mkdir -p data/local
if [[ ! -f data/local/lexicon.txt || data/local/lexicon.txt -ot "$lexicon_file" ]]; then
  echo ---------------------------------------------------------------------
  echo "Preparing lexicon in data/local on" `date`
  echo ---------------------------------------------------------------------
  local/prepare_lexicon.pl \
    $lexiconFlags $lexicon_file data/local || exit 1
fi

mkdir -p data/lang
if [[ ! -f data/lang/L.fst || data/lang/L.fst -ot data/local/lexicon.txt ]]; then
  echo ---------------------------------------------------------------------
  echo "Creating L.fst etc in data/lang on" `date`
  echo ---------------------------------------------------------------------
  utils/prepare_lang.sh \
    --share-silence-phones true \
    data/local $oovSymbol data/local/tmp.lang data/lang || exit 1
fi

if [[ ! -f data/train/wav.scp || data/train/wav.scp -ot "$train_data_dir" ]]; then
  echo ---------------------------------------------------------------------
  echo "Preparing acoustic training lists in data/train on" `date`
  echo ---------------------------------------------------------------------
  mkdir -p data/train
  local/prepare_acoustic_training_data.pl \
    --vocab data/local/lexicon.txt --fragmentMarkers \-\*\~ \
    $train_data_dir data/train > data/train/skipped_utts.log || exit 1
fi

if [[ ! -f data/dev2h/wav.scp || data/dev2h/wav.scp -ot "$dev2h_data_dir" ]]; then
  echo ---------------------------------------------------------------------
  echo "Preparing dev2h data lists in data/dev2h on" `date`
  echo ---------------------------------------------------------------------
  mkdir -p data/dev2h
  local/prepare_acoustic_training_data.pl \
    --fragmentMarkers \-\*\~ \
    $dev2h_data_dir data/dev2h > data/dev2h/skipped_utts.log || exit 1
fi


if [[ ! -f data/dev2h/glm || data/dev2h/glm -ot "$glmFile" ]]; then
  echo ---------------------------------------------------------------------
  echo "Preparing dev2h stm files in data/dev2h on" `date`
  echo ---------------------------------------------------------------------
  local/prepare_stm.pl --fragmentMarkers \-\*\~ data/dev2h || exit 1
  cp $glmFile data/dev2h/glm
fi

# We will simply override the default G.fst by the G.fst generated using SRILM
if [[ ! -f data/srilm/lm.gz || data/srilm/lm.gz -ot data/train/text ]]; then
  echo ---------------------------------------------------------------------
  echo "Training SRILM language models on" `date`
  echo ---------------------------------------------------------------------
  local/train_lms_srilm.sh --dev-text data/dev2h/text --train-text data/train/text data data/srilm 
fi
if [[ ! -f data/lang/G.fst || data/lang/G.fst -ot data/srilm/lm.gz ]]; then
  echo ---------------------------------------------------------------------
  echo "Creating G.fst on " `date`
  echo ---------------------------------------------------------------------
  local/arpa2G.sh data/srilm/lm.gz data/lang data/lang
fi
  

if [ ! -f data/dev2h/.kws.done ]; then
  if [[ $subset_ecf ]] ; then
    local/kws_setup.sh --case-insensitive $case_insensitive --subset-ecf $dev2h_data_list $ecf_file $kwlist_file $rttm_file data/lang data/dev2h || exit 1
  else
    local/kws_setup.sh --case-insensitive $case_insensitive $ecf_file $kwlist_file $rttm_file data/lang data/dev2h || exit 1
  fi
  touch data/dev2h/.kws.done
fi


if [ ! -f data/train/.plp.done ]; then
  echo ---------------------------------------------------------------------
  echo "Starting plp feature extraction in plp on" `date`
  echo ---------------------------------------------------------------------
  steps/make_plp.sh \
    --cmd "$train_cmd" --nj $train_nj \
    data/train exp/make_plp/train plp || exit 1
  steps/compute_cmvn_stats.sh \
    data/train exp/make_plp/train plp || exit 1
# In case plp extraction failed on some utterance, delist them
  utils/fix_data_dir.sh data/train
  touch data/train/.plp.done
fi

if [ ! -f data/dev2h/.plp.done ]; then
  steps/make_plp.sh \
    --cmd "$train_cmd" --nj $decode_nj \
    data/dev2h exp/make_plp/dev2h plp || exit 1
  steps/compute_cmvn_stats.sh \
    data/dev2h exp/make_plp/dev2h plp || exit 1
  # In case plp extraction failed on some utterance, delist them
  utils/fix_data_dir.sh data/dev2h
  touch data/dev2h/.plp.done
fi
mkdir -p exp

if [ ! -f data/train_sub3/.done ]; then
  echo ---------------------------------------------------------------------
  echo "Subsetting monophone training data in data/train_sub[123] on" `date`
  echo ---------------------------------------------------------------------
  numutt=`cat data/train/feats.scp | wc -l`;
  utils/subset_data_dir.sh data/train  5000 data/train_sub1 || exit 1
  if [ $numutt -gt 10000 ] ; then
    utils/subset_data_dir.sh data/train 10000 data/train_sub2 || exit 1
  else
    (cd data; ln -s train train_sub2 ) || exit 1
  fi
  if [ $numutt -gt 20000 ] ; then
    utils/subset_data_dir.sh data/train 20000 data/train_sub3 || exit 1
  else
    (cd data; ln -s train train_sub3 ) || exit 1
  fi

  touch data/train_sub3/.done
fi

if [ ! -f exp/mono/.done ]; then
  echo ---------------------------------------------------------------------
  echo "Starting (small) monophone training in exp/mono on" `date`
  echo ---------------------------------------------------------------------
  steps/train_mono.sh \
    --boost-silence $boost_sil --nj 8 --cmd "$train_cmd" \
    data/train_sub1 data/lang exp/mono || exit 1
  touch exp/mono/.done
fi

if [ ! -f exp/tri1/.done ]; then
  echo ---------------------------------------------------------------------
  echo "Starting (small) triphone training in exp/tri1 on" `date`
  echo ---------------------------------------------------------------------
  steps/align_si.sh \
    --boost-silence $boost_sil --nj 12 --cmd "$train_cmd" \
    data/train_sub2 data/lang exp/mono exp/mono_ali_sub2 || exit 1
  steps/train_deltas.sh \
    --boost-silence $boost_sil --cmd "$train_cmd" \
    $numLeavesTri1 $numGaussTri1 data/train_sub2 data/lang exp/mono_ali_sub2 exp/tri1 || exit 1
  touch exp/tri1/.done
fi

decode_si() { 
  dir=$1
  if [ ! -f $dir/decode_dev2h/.done ]; then
    echo ---------------------------------------------------------------
    echo "Spawning decoding with triphone models in $dir on" `date`
    echo ---------------------------------------------------------------
    mkdir -p $dir/graph
    utils/mkgraph.sh data/lang $dir $dir/graph &> $dir/mkgraph.log || exit 1;
    mkdir -p $dir/decode_dev2h
    steps/decode.sh --nj $decode_nj --cmd "$decode_cmd" \
       --num-threads 6 --parallel-opts "-pe smp 6 -l mem_free=4G,ram_free=0.7G" \
       $dir/graph data/dev2h $dir/decode_dev2h &> $dir/decode_dev2h.log || exit 1;

    local/kws_search.sh --cmd "$decode_cmd" --duptime $duptime \
        data/lang data/dev2h $dir/decode_dev2h
    touch $dir/decode_dev2h/.done
    echo "See $dir/mkgraph.log and $dir/decode_dev2h.log for decoding outcomes"
  fi
}

decode_si exp/tri1 &
sleep 5;  # Let any "start-up error" messages from the subshell get logged

echo ---------------------------------------------------------------------
echo "Starting (medium) triphone training in exp/tri2 on" `date`
echo ---------------------------------------------------------------------
if [ ! -f exp/tri2/.done ]; then
  steps/align_si.sh \
    --boost-silence $boost_sil --nj 24 --cmd "$train_cmd" \
    data/train_sub3 data/lang exp/tri1 exp/tri1_ali_sub3 || exit 1
  steps/train_deltas.sh \
    --boost-silence $boost_sil --cmd "$train_cmd" \
    $numLeavesTri2 $numGaussTri2 data/train_sub3 data/lang exp/tri1_ali_sub3 exp/tri2 || exit 1
  touch exp/tri2/.done
fi

decode_si exp/tri2 &
sleep 5; 

echo ---------------------------------------------------------------------
echo "Starting (full) triphone training in exp/tri3 on" `date`
echo ---------------------------------------------------------------------
if [ ! -f exp/tri3/.done ]; then
  steps/align_si.sh \
    --boost-silence $boost_sil --nj $train_nj --cmd "$train_cmd" \
    data/train data/lang exp/tri2 exp/tri2_ali || exit 1
  steps/train_deltas.sh \
    --boost-silence $boost_sil --cmd "$train_cmd" \
    $numLeavesTri3 $numGaussTri3 data/train data/lang exp/tri2_ali exp/tri3 || exit 1
  touch exp/tri3/.done
fi

decode_si exp/tri3 &
sleep 5

echo ---------------------------------------------------------------------
echo "Starting (lda_mllt) triphone training in exp/tri4 on" `date`
echo ---------------------------------------------------------------------
if [ ! -f exp/tri4/.done ]; then
  steps/align_si.sh \
    --boost-silence $boost_sil --nj $train_nj --cmd "$train_cmd" \
    data/train data/lang exp/tri3 exp/tri3_ali || exit 1
  steps/train_lda_mllt.sh \
    --boost-silence $boost_sil --cmd "$train_cmd" \
    $numLeavesMLLT $numGaussMLLT data/train data/lang exp/tri3_ali exp/tri4 || exit 1
  touch exp/tri4/.done
fi
decode_si exp/tri4 &

echo ---------------------------------------------------------------------
echo "Starting (SAT) triphone training in exp/tri5 on" `date`
echo ---------------------------------------------------------------------

if [ ! -f exp/tri5/.done ]; then
  steps/align_si.sh \
    --boost-silence $boost_sil --nj $train_nj --cmd "$train_cmd" \
    data/train data/lang exp/tri4 exp/tri4_ali || exit 1
  steps/train_sat.sh \
    --boost-silence $boost_sil --cmd "$train_cmd" \
    $numLeavesSAT $numGaussSAT data/train data/lang exp/tri4_ali exp/tri5 || exit 1
  touch exp/tri5/.done
fi

(
  if [ ! -f exp/tri5/decode_dev2h/.done ]; then
    echo ---------------------------------------------------------------------
    echo "Spawning decoding with SAT models  on" `date`
    echo ---------------------------------------------------------------------
    mkdir -p exp/tri5/graph
    utils/mkgraph.sh \
        data/lang exp/tri5 exp/tri5/graph &> exp/tri5/mkgraph.log
    mkdir -p exp/tri5/decode_dev2h
    touch exp/tri5/decode_dev2h.started # A signal to the SGMM2 decoding step
    steps/decode_fmllr.sh --nj $decode_nj --cmd "$decode_cmd" --num-threads 6 \
        --parallel-opts "-pe smp 6 -l mem_free=4G,ram_free=0.7G" \
      exp/tri5/graph data/dev2h exp/tri5/decode_dev2h &> exp/tri5/decode_dev2h.log 
    touch exp/tri5/decode_dev2h/.done
  fi

  if [ ! -f exp/tri5/decode_dev2h/kws/.done ]; then
    local/kws_search.sh --cmd "$decode_cmd" --duptime $duptime \
        data/lang data/dev2h exp/tri5/decode_dev2h.si
    local/kws_search.sh --cmd "$decode_cmd" --duptime $duptime \
        data/lang data/dev2h exp/tri5/decode_dev2h
    touch exp/tri5/decode_dev2h/kws/.done 
  fi
) &
sleep 5; # Let any "start-up error" messages from the subshell get logged


################################################################################
# Ready to start SGMM training
################################################################################

if [ ! -f exp/tri5_ali/.done ]; then
  echo ---------------------------------------------------------------------
  echo "Starting exp/tri5_ali on" `date`
  echo ---------------------------------------------------------------------
  steps/align_fmllr.sh \
    --boost-silence $boost_sil --nj $train_nj --cmd "$train_cmd" \
    data/train data/lang exp/tri5 exp/tri5_ali || exit 1
  touch exp/tri5_ali/.done
fi

if [ ! -f exp/ubm5/.done ]; then
  echo ---------------------------------------------------------------------
  echo "Starting exp/ubm5 on" `date`
  echo ---------------------------------------------------------------------
  steps/train_ubm.sh \
    --cmd "$train_cmd" \
    $numGaussUBM data/train data/lang exp/tri5_ali exp/ubm5 || exit 1
  touch exp/ubm5/.done
fi

if [ ! -f exp/sgmm5/.done ]; then
  echo ---------------------------------------------------------------------
  echo "Starting exp/sgmm5 on" `date`
  echo ---------------------------------------------------------------------
  steps/train_sgmm2_group.sh \
    --cmd "$train_cmd" --group 3 --parallel-opts "-l mem_free=6G,ram_free=2G" \
    $numLeavesSGMM $numGaussSGMM data/train data/lang exp/tri5_ali exp/ubm5/final.ubm exp/sgmm5 || exit 1
  touch exp/sgmm5/.done
fi

################################################################################
# Ready to decode with SGMM2 models
################################################################################


(
  if [ ! -f exp/sgmm5/decode_dev2h/.done ]; then
    echo ---------------------------------------------------------------------
    echo "Spawning exp/sgmm5/decode_dev2h_fmllr on" `date`
    echo ---------------------------------------------------------------------
    echo "exp/sgmm5/decode_dev2h will wait on tri5 decode if necessary"
    while [ ! -f exp/tri5/decode_dev2h_fmllr/.done ]; do sleep 30; done
    mkdir -p exp/sgmm5/graph
    utils/mkgraph.sh \
        data/lang exp/sgmm5 exp/sgmm5/graph &> exp/sgmm5/mkgraph.log

    steps/decode_sgmm2.sh --use-fmllr true --nj $decode_nj --cmd "$decode_cmd" \
        --num-threads 6 --parallel-opts "-pe smp 6 -l mem_free=5G,ram_free=0.8G" \
        --transform-dir exp/tri5/decode_dev2h \
        exp/sgmm5/graph data/dev2h/ exp/sgmm5/decode_dev2h_fmllr &> exp/sgmm5/decode_dev2h_fmllr.log
    touch exp/sgmm5/decode_dev2h_fmllr/.done
  fi

  if [ ! -f exp/sgmm5/decode_dev2h_fmllr/kws/.done ]; then
    echo ---------------------------------------------------------------------
    echo "Starting exp/sgmm5/decode_dev2h_fmllr/kws on" `date`
    echo ---------------------------------------------------------------------
    local/kws_search.sh --cmd "$decode_cmd" --duptime $duptime \
        data/lang data/dev2h exp/sgmm5/decode_dev2h_fmllr
    touch exp/sgmm5/decode_dev2h_fmllr/kws/.done
  fi
) &

sleep 10; # Let any "start-up error" messages from the subshell get logged
echo "See exp/sgmm5/mkgraph.log, exp/sgmm5/decode_dev2h.log and exp/sgmm5/decode_dev2h_fmllr.log for decoding outcomes"

################################################################################
# Ready to start discriminative SGMM training
################################################################################

if [ ! -f exp/sgmm5_ali/.done ]; then
  echo ---------------------------------------------------------------------
  echo "Starting exp/sgmm5_ali on" `date`
  echo ---------------------------------------------------------------------
  steps/align_sgmm2.sh \
    --nj $train_nj --cmd "$train_cmd" --transform-dir exp/tri5_ali --use-graphs true --use-gselect true \
    data/train data/lang exp/sgmm5 exp/sgmm5_ali || exit 1
  touch exp/sgmm5_ali/.done
fi

if [ ! -f exp/sgmm5_denlats/.done ]; then
  echo ---------------------------------------------------------------------
  echo "Starting exp/sgmm5_denlats on" `date`
  echo ---------------------------------------------------------------------
  steps/make_denlats_sgmm2.sh \
    --num-threads 4 --parallel-opts "-pe smp 4" --cmd "queue.pl -l mem_free=2G,ram_free=0.8G" \
    --nj $train_nj --sub-split $train_nj  \
    --beam 10.0 --lattice-beam 6 --cmd "$decode_cmd" --transform-dir exp/tri5_ali \
    data/train data/lang exp/sgmm5_ali exp/sgmm5_denlats || exit 1
  touch exp/sgmm5_denlats/.done
fi

if [ ! -f exp/sgmm5_mmi_b0.1/.done ]; then
  echo ---------------------------------------------------------------------
  echo "Starting exp/sgmm5_mmi_b0.1 on" `date`
  echo ---------------------------------------------------------------------
  steps/train_mmi_sgmm2.sh \
    --cmd "queue.pl -l mem_free=4G,ram_free=4.5G" \
    --zero-if-disjoint true --transform-dir exp/tri5_ali --boost 0.1 \
    data/train data/lang exp/sgmm5_ali exp/sgmm5_denlats \
    exp/sgmm5_mmi_b0.1 || exit 1
  touch exp/sgmm5_mmi_b0.1/.done
fi

################################################################################
# Ready to decode with discriminative SGMM2 models
################################################################################

wait $sgmm5decode; # Need lattices from the corresponding SGMM decoding passes


echo ---------------------------------------------------------------------
echo "Starting exp/sgmm5_mmi_b0.1/decode_dev2h[_fmllr] on" `date`
echo ---------------------------------------------------------------------
for iter in 1 2 3 4; do
  if [ ! -f exp/sgmm5_mmi_b0.1/decode_dev2h_fmllr_it$iter/.done ]; then
    echo "Waiting for exp/sgmm5/decode_dev2h_fmllr_it$iter/.done if necessary"
    while [ ! -f exp/sgmm5/decode_dev2h_fmllr_it$iter/.done ]; do sleep 30; done
    steps/decode_sgmm2_rescore.sh \
        --cmd "$decode_cmd" --iter $iter --transform-dir exp/tri5/decode_dev2h \
        data/lang data/dev2h exp/sgmm5/decode_dev2h_fmllr exp/sgmm5_mmi_b0.1/decode_dev2h_fmllr_it$iter
    touch exp/sgmm5_mmi_b0.1/decode_dev2h_fmllr_it$iter/.done
  fi
  if [ ! -f exp/sgmm5_mmi_b0.1/decode_dev2h_fmllr_it$iter/kws/.done ]; then
    local/kws_search.sh --cmd "$decode_cmd" --duptime $duptime \
        data/lang data/dev2h exp/sgmm5_mmi_b0.1/decode_dev2h_fmllr_it$iter
    touch exp/sgmm5_mmi_b0.1/decode_dev2h_fmllr_it$iter/kws/.done
  fi
done

wait

echo ---------------------------------------------------------------------
echo "Finished successfully on" `date`
echo ---------------------------------------------------------------------

exit 0
