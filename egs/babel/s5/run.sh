#!/bin/bash

# System and data directories
#SCRIPT=$(readlink -f $0)
#SysDir=`dirname $SCRIPT`
SysDir=`pwd`
echo $SysDir

#keyword search default
glmFile=conf/glm
duptime=0.5
case_insensitive=false

# Lexicon and Language Model parameters
oovSymbol="<unk>"
lexiconFlags="-oov <unk>"

# Include the checkpointing facility
. ./local/CHECKPOINT.sh

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
[ -f ./cmd.sh ] && . ./cmd.sh; # source the path.
. parse_options.sh || exit 1;

configfile=$1
[ -f $configfile ] && . $configfile 
[ -f ./local.conf ] && . ./local.conf

#Preparing dev and train directories
if [[ ! -z  "$train_data_list" ]] ; then
    echo ---------------------------------------------------------------------
    echo "Subsetting the TRAIN set"
    echo ---------------------------------------------------------------------

    local/make_corpus_subset.sh $train_data_dir $train_data_list ./data/raw_train_data || exit 1
    train_data_dir=`readlink -f ./data/raw_train_data`

    nj_max=`cat $train_data_list | wc -l`
    if [[ "$nj_max" -lt "$train_nj" ]] ; then
        echo "The maximum reasonable number of jobs is $nj_max (you have $train_nj)! (The training and decoding process has file-granularity)"
        train_nj=$nj_max
    fi
fi

if [[ ! -z $dev_data_list ]] ; then
    echo ---------------------------------------------------------------------
    echo "Subsetting the DEV set"
    echo ---------------------------------------------------------------------

    local/make_corpus_subset.sh $dev_data_dir $dev_data_list ./data/raw_dev_data || exit 1
    dev_data_dir=`readlink -f ./data/raw_dev_data`

    nj_max=`cat $dev_data_list | wc -l`
    if [[ "$nj_max" -lt "$decode_nj" ]] ; then
        echo "The maximum reasonable number of jobs is $nj_max -- you have $decode_nj! (The training and decoding process has file-granularity)"
        decode_nj=$nj_max
    fi
fi

if [[ $filter_lexicon ]]; then
    echo ---------------------------------------------------------------------
    echo "Subsetting the LEXICON"
    echo ---------------------------------------------------------------------

    lexicon_dir=./data/raw_lex_data
    mkdir -p $lexicon_dir
    local/make_lexicon_subset.sh $train_data_dir/transcription \
        $lexicon_file $lexicon_dir/lexicon.txt || exit 1
    lexicon_file=$lexicon_dir/lexicon.txt
fi


echo ---------------------------------------------------------------------
echo "Preparing lexicon in data/local on" `date`
echo ---------------------------------------------------------------------
mkdir -p data/local
local/prepare_lexicon.pl \
    $lexiconFlags $lexicon_file data/local || exit 1

echo ---------------------------------------------------------------------
echo "Creating L.fst etc in data/lang on" `date`
echo ---------------------------------------------------------------------
mkdir -p data/lang
utils/prepare_lang.sh \
    --share-silence-phones true \
    data/local $oovSymbol data/local/tmp.lang data/lang || exit 1

echo ---------------------------------------------------------------------
echo "Preparing acoustic training lists in data/train on" `date`
echo ---------------------------------------------------------------------
mkdir -p data/train
local/prepare_acoustic_training_data.pl \
    --vocab data/local/lexicon.txt --fragmentMarkers \-\*\~ \
    $train_data_dir data/train > data/train/skipped_utts.log || exit 1

echo ---------------------------------------------------------------------
echo "Preparing dev data lists in data/dev on" `date`
echo ---------------------------------------------------------------------
mkdir -p data/dev
local/prepare_acoustic_training_data.pl \
    --fragmentMarkers \-\*\~ \
    $dev_data_dir data/dev > data/dev/skipped_utts.log || exit 1

echo ---------------------------------------------------------------------
echo "Preparing dev stm files in data/dev on" `date`
echo ---------------------------------------------------------------------
local/prepare_stm.pl --fragmentMarkers \-\*\~ data/dev || exit 1
cp $glmFile data/dev/glm

echo ---------------------------------------------------------------------
echo "Creating a basic G.fst in data/lang on" `date`
echo ---------------------------------------------------------------------
# We will simply override the default G.fst by the G.fst generated using SRILM
local/train_lms_srilm.sh data data/srilm 
local/arpa2G.sh data/srilm/lm.gz data/lang data/lang

if [[ $subset_ecf ]] ; then
    local/kws_setup.sh --case-insensitive $case_insensitive --subset-ecf $dev_data_list $ecf_file $kwlist_file $rttm_file data/lang data/dev || exit 1
else
    local/kws_setup.sh --case-insensitive $case_insensitive $ecf_file $kwlist_file $rttm_file data/lang data/dev || exit 1
fi

cd $SysDir
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

steps/make_plp.sh \
    --cmd "$train_cmd" --nj $decode_nj \
    data/dev exp/make_plp/dev plp || exit 1
steps/compute_cmvn_stats.sh \
    data/dev exp/make_plp/dev plp || exit 1
# In case plp extraction failed on some utterance, delist them
utils/fix_data_dir.sh data/dev
mkdir -p exp

echo ---------------------------------------------------------------------
echo "Subsetting monophone training data in data/train_sub[123] on" `date`
echo ---------------------------------------------------------------------
utils/subset_data_dir.sh data/train  5000 data/train_sub1 || exit 1
utils/subset_data_dir.sh data/train 10000 data/train_sub2 || exit 1
utils/subset_data_dir.sh data/train 20000 data/train_sub3 || exit 1

echo ---------------------------------------------------------------------
echo "Starting (small) monophone training in exp/mono on" `date`
echo ---------------------------------------------------------------------
steps/train_mono.sh \
    --boost-silence 1.5 --nj 8 --cmd "$train_cmd" \
    data/train_sub1 data/lang exp/mono || exit 1

echo ---------------------------------------------------------------------
echo "Starting (small) triphone training in exp/tri1 on" `date`
echo ---------------------------------------------------------------------
steps/align_si.sh \
    --boost-silence 1.5 --nj 12 --cmd "$train_cmd" \
    data/train_sub2 data/lang exp/mono exp/mono_ali_sub2 || exit 1
steps/train_deltas.sh \
    --boost-silence 1.5 --cmd "$train_cmd" \
    $numLeavesTri1 $numGaussTri1 data/train_sub2 data/lang exp/mono_ali_sub2 exp/tri1 || exit 1

echo ---------------------------------------------------------------------
echo "Spawning decoding with (small) triphone models in exp/tri1 on" `date`
echo ---------------------------------------------------------------------
(
    mkdir -p exp/tri1/graph
    utils/mkgraph.sh data/lang exp/tri1 exp/tri1/graph &> exp/tri1/mkgraph.log
    mkdir -p exp/tri1/decode
    steps/decode.sh --nj $decode_nj --cmd "$decode_cmd" \
        exp/tri1/graph data/dev exp/tri1/decode &> exp/tri1/decode.log

    local/kws_search.sh --cmd "$decode_cmd" --duptime $duptime \
        data/lang data/dev exp/tri1/decode
) &
tri1decode=$!; # Grab the PID of the subshell
sleep 5; # Let any "start-up error" messages from the subshell get logged
echo "See exp/tri1/mkgraph.log and exp/tri1/decode.log for decoding outcomes"

echo ---------------------------------------------------------------------
echo "Starting (medium) triphone training in exp/tri2 on" `date`
echo ---------------------------------------------------------------------
steps/align_si.sh \
    --boost-silence 1.5 --nj 24 --cmd "$train_cmd" \
    data/train_sub3 data/lang exp/tri1 exp/tri1_ali_sub3 || exit 1
steps/train_deltas.sh \
    --boost-silence 1.5 --cmd "$train_cmd" \
    $numLeavesTri2 $numGaussTri2 data/train_sub3 data/lang exp/tri1_ali_sub3 exp/tri2 || exit 1

echo ---------------------------------------------------------------------
echo "Spawning decoding with (medium) triphone models in exp/tri2 on" `date`
echo ---------------------------------------------------------------------
(
    mkdir -p exp/tri2/graph
    utils/mkgraph.sh data/lang exp/tri2 exp/tri2/graph &> exp/tri2/mkgraph.log
    mkdir -p exp/tri2/decode
    steps/decode.sh --nj $decode_nj --cmd "$decode_cmd" \
        exp/tri2/graph data/dev exp/tri2/decode &> exp/tri2/decode.log

    local/kws_search.sh --cmd "$decode_cmd" --duptime $duptime \
        data/lang data/dev exp/tri2/decode
) &
tri2decode=$!; # Grab the PID of the subshell
sleep 5; # Let any "start-up error" messages from the subshell get logged
echo "See exp/tri2/mkgraph.log and exp/tri2/decode.log for decoding outcomes"

echo ---------------------------------------------------------------------
echo "Starting (full) triphone training in exp/tri3 on" `date`
echo ---------------------------------------------------------------------
steps/align_si.sh \
    --boost-silence 1.5 --nj $train_nj --cmd "$train_cmd" \
    data/train data/lang exp/tri2 exp/tri2_ali || exit 1
steps/train_deltas.sh \
    --boost-silence 1.5 --cmd "$train_cmd" \
    $numLeavesTri3 $numGaussTri3 data/train data/lang exp/tri2_ali exp/tri3 || exit 1

echo ---------------------------------------------------------------------
echo "Spawning decoding with full triphone models in exp/tri3 on" `date`
echo ---------------------------------------------------------------------
(
    mkdir -p exp/tri3/graph
    utils/mkgraph.sh data/lang exp/tri3 exp/tri3/graph &> exp/tri3/mkgraph.log
    mkdir -p exp/tri3/decode
    steps/decode.sh --nj $decode_nj --cmd "$decode_cmd" \
        exp/tri3/graph data/dev exp/tri3/decode &> exp/tri3/decode.log

    local/kws_search.sh --cmd "$decode_cmd" --duptime $duptime \
        data/lang data/dev exp/tri3/decode
) &
tri3decode=$!; # Grab the PID of the subshell
sleep 5; # Let any "start-up error" messages from the subshell get logged
echo "See exp/tri3/mkgraph.log and exp/tri3/decode.log for decoding outcomes"

echo ---------------------------------------------------------------------
echo "Starting (lda_mllt) triphone training in exp/tri4 on" `date`
echo ---------------------------------------------------------------------
steps/align_si.sh \
    --boost-silence 1.5 --nj $train_nj --cmd "$train_cmd" \
    data/train data/lang exp/tri3 exp/tri3_ali || exit 1
steps/train_lda_mllt.sh \
    --boost-silence 1.5 --cmd "$train_cmd" \
    $numLeavesMLLT $numGaussMLLT data/train data/lang exp/tri3_ali exp/tri4 || exit 1

echo ---------------------------------------------------------------------
echo "Spawning decoding with lda_mllt models in exp/tri4 on" `date`
echo ---------------------------------------------------------------------
(
    mkdir -p exp/tri4/graph
    utils/mkgraph.sh \
        data/lang exp/tri4 exp/tri4/graph &> exp/tri4/mkgraph.log
    mkdir -p exp/tri4/decode
    steps/decode.sh --nj $decode_nj --cmd "$decode_cmd" \
        exp/tri4/graph data/dev exp/tri4/decode &> exp/tri4/decode.log

    local/kws_search.sh --cmd "$decode_cmd" --duptime $duptime \
        data/lang data/dev exp/tri4/decode
) &
tri4decode=$!; # Grab the PID of the subshell
sleep 5; # Let any "start-up error" messages from the subshell get logged
echo "See exp/tri4/mkgraph.log and exp/tri4/decode.log for decoding outcomes"

echo ---------------------------------------------------------------------
echo "Starting (SAT) triphone training in exp/tri5 on" `date`
echo ---------------------------------------------------------------------

steps/align_si.sh \
    --boost-silence 1.5 --nj $train_nj --cmd "$train_cmd" \
    data/train data/lang exp/tri4 exp/tri4_ali || exit 1
steps/train_sat.sh \
    --boost-silence 1.5 --cmd "$train_cmd" \
    $numLeavesSAT $numGaussSAT data/train data/lang exp/tri4_ali exp/tri5 || exit 1

echo ---------------------------------------------------------------------
echo "Spawning decoding with SAT models  on" `date`
echo ---------------------------------------------------------------------
(
    mkdir -p exp/tri5/graph
    utils/mkgraph.sh \
        data/lang exp/tri5 exp/tri5/graph &> exp/tri5/mkgraph.log
    mkdir -p exp/tri5/decode
    touch exp/tri5/decode.started # A signal to the SGMM2 decoding step
    steps/decode_fmllr.sh --nj $decode_nj --cmd "$decode_cmd" \
        exp/tri5/graph data/dev exp/tri5/decode &> exp/tri5/decode.log \
    && touch exp/tri5/decode.finished # so SGMM2 decoding may proceed

    local/kws_search.sh --cmd "$decode_cmd" --duptime $duptime \
        data/lang data/dev exp/tri5/decode.si
    local/kws_search.sh --cmd "$decode_cmd" --duptime $duptime \
        data/lang data/dev exp/tri5/decode
) &
tri5decode=$!; # Grab the PID of the subshell; needed for SGMM2 decoding
sleep 30; # Let any "start-up error" messages from the subshell get logged
echo "See exp/tri5/mkgraph.log and exp/tri5/decode.log for decoding outcomes"

################################################################################
# Ready to start SGMM training
################################################################################

echo ---------------------------------------------------------------------
echo "Starting exp/ubm5 on" `date`
echo ---------------------------------------------------------------------
steps/align_fmllr.sh \
    --boost-silence 1.5 --nj $train_nj --cmd "$train_cmd" \
    data/train data/lang exp/tri5 exp/tri5_ali || exit 1


steps/train_ubm.sh \
    --cmd "$train_cmd" \
    $numGaussUBM data/train data/lang exp/tri5_ali exp/ubm5 || exit 1

echo ---------------------------------------------------------------------
echo "Starting exp/sgmm5 on" `date`
echo ---------------------------------------------------------------------
steps/train_sgmm2.sh \
    --cmd "$train_cmd" \
    $numLeavesSGMM $numGaussSGMM data/train data/lang exp/tri5_ali exp/ubm5/final.ubm exp/sgmm5 || exit 1

################################################################################
# Ready to decode with SGMM2 models
################################################################################

echo ---------------------------------------------------------------------
echo "Spawning exp/sgmm5/decode[_fmllr] on" `date`
echo ---------------------------------------------------------------------
echo "exp/sgmm5/decode will wait on PID $tri5decode if necessary"
wait $tri5decode; # Need lattices from the corresponding SGMM decoding passes
(
    sleep 5; # Let the status message after the subshell get logged
## The next (now commented) block should ensure we starting decoding of sgmm5 only after
## the tri5 decoding finishes. The same can be achieved by "wait"int for tri5decode pid
#    while [ ! -f exp/tri5/decode.started -o ! -f exp/tri5/decode.finished ]; do
#        echo "exp/sgmm5/decode is waiting on SAT decoding ..." `date`
#        sleep 5
#    done
#    while [ exp/tri5/decode.finished -ot exp/tri5/decode.started ]; do
#        echo "exp/tri5/decode.finished is older than exp/tri5/decode.started"; \
#        ls -lt exp/tri5/decode.finished exp/tri5/decode.started; \
#        echo "Perhaps SAT decoding was restarted and is still running?"; \
#        echo "exp/sgmm5/decode is still waiting on SAT decoding ..." `date`
#        sleep 5
#    done
#    rm exp/tri5/decode.started exp/tri5/decode.finished
    mkdir -p exp/sgmm5/graph
    utils/mkgraph.sh \
        data/lang exp/sgmm5 exp/sgmm5/graph &> exp/sgmm5/mkgraph.log
    mkdir -p exp/sgmm5/decode
    steps/decode_sgmm2.sh \
        --nj $decode_nj --cmd "$decode_cmd" --transform-dir exp/tri5/decode \
        exp/sgmm5/graph data/dev/ exp/sgmm5/decode &> exp/sgmm5/decode.log
    steps/decode_sgmm2.sh --use-fmllr true --nj $decode_nj --cmd "$decode_cmd" \
        --transform-dir exp/tri5/decode \
        exp/sgmm5/graph data/dev/ exp/sgmm5/decode_fmllr &> exp/sgmm5/decode_fmllr.log

    local/kws_search.sh --cmd "$decode_cmd" --duptime $duptime \
        data/lang data/dev exp/sgmm5/decode
    local/kws_search.sh --cmd "$decode_cmd" --duptime $duptime \
        data/lang data/dev exp/sgmm5/decode_fmllr
) &
sgmm5decode=$!; # Grab the PID of the subshell; needed for MMI rescoring
sleep 30; # Let any "start-up error" messages from the subshell get logged
echo "See exp/sgmm5/mkgraph.log, exp/sgmm5/decode.log and exp/sgmm5/decode_fmllr.log for decoding outcomes"

################################################################################
# Ready to start discriminative SGMM training
################################################################################

echo ---------------------------------------------------------------------
echo "Starting exp/sgmm5_ali on" `date`
echo ---------------------------------------------------------------------
steps/align_sgmm2.sh \
    --nj $train_nj --cmd "$train_cmd" --transform-dir exp/tri5_ali --use-graphs true --use-gselect true \
    data/train data/lang exp/sgmm5 exp/sgmm5_ali || exit 1

echo ---------------------------------------------------------------------
echo "Starting exp/sgmm5_denlats on" `date`
echo ---------------------------------------------------------------------
steps/make_denlats_sgmm2.sh \
    --nj $train_nj --sub-split $train_nj  \
    --beam 10.0 --lattice-beam 6 --cmd "$decode_cmd" --transform-dir exp/tri5_ali \
    data/train data/lang exp/sgmm5_ali exp/sgmm5_denlats || exit 1

echo ---------------------------------------------------------------------
echo "Starting exp/sgmm5_mmi_b0.1 on" `date`
echo ---------------------------------------------------------------------
steps/train_mmi_sgmm2.sh \
    --cmd "$decode_cmd" --zero-if-disjoint true --transform-dir exp/tri5_ali --boost 0.1 \
    data/train data/lang exp/sgmm5_ali exp/sgmm5_denlats \
    exp/sgmm5_mmi_b0.1 || exit 1

################################################################################
# Ready to decode with discriminative SGMM2 models
################################################################################

echo "exp/sgmm5_mmi_b0.1/decode will wait on PID $sgmm5decode if necessary"
wait $sgmm5decode; # Need lattices from the corresponding SGMM decoding passes
echo ---------------------------------------------------------------------
echo "Starting exp/sgmm5_mmi_b0.1/decode[_fmllr] on" `date`
echo ---------------------------------------------------------------------
for iter in 1 2 3 4; do
    steps/decode_sgmm2_rescore.sh \
        --cmd "$decode_cmd" --iter $iter --transform-dir exp/tri5/decode \
        data/lang data/dev exp/sgmm5/decode exp/sgmm5_mmi_b0.1/decode_it$iter
    steps/decode_sgmm2_rescore.sh \
        --cmd "$decode_cmd" --iter $iter --transform-dir exp/tri5/decode \
        data/lang data/dev exp/sgmm5/decode_fmllr exp/sgmm5_mmi_b0.1/decode_fmllr_it$iter

    local/kws_search.sh --cmd "$decode_cmd" --duptime $duptime \
        data/lang data/dev exp/sgmm5_mmi_b0.1/decode_it$iter
    local/kws_search.sh --cmd "$decode_cmd" --duptime $duptime \
        data/lang data/dev exp/sgmm5_mmi_b0.1/decode_fmllr_it$iter
done

wait $tri1decode
wait $tri2decode
wait $tri3decode
wait $tri4decode

# No need to wait on $tri5decode ---> $sgmm5decode ---> sgmm5_mmi_b0.1decode

echo ---------------------------------------------------------------------
echo "Finished successfully on" `date`
echo ---------------------------------------------------------------------

exit 0
