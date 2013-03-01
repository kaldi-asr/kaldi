#!/bin/bash


# System and data directories
#SCRIPT=$(readlink -f $0)
#SysDir=`dirname $SCRIPT`
SysDir=`pwd`
echo $SysDir


# Lexicon and Language Model parameters
oovSymbol="<unk>"
lexiconFlags="-oov <unk>"

duptime=0.5
case_insensitive=false

# Scoring protocols (dummy GLM file to appease the scoring script)
glmFile=`readlink -f ./conf/glm`

. ./local/CHECKPOINT.sh

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
[ -f ./cmd.sh ] && . ./cmd.sh; # source the path.
. parse_options.sh || exit 1;

configfile=$1
[ -f $configfile ] && . $configfile 
[ -f ./local.conf ] && . ./local.conf

#Preparing test.pem directories


if [[ ! -z $test_data_list ]] ; then
    echo "Subsetting the TEST set"

    local/make_corpus_subset.sh $test_data_dir $test_data_list ./data/raw_test_data || exit 1
    test_data_dir=`readlink -f ./data/raw_test_data`
fi


echo --------------------------------------------------------------------
echo "Preparing test data lists in data/test.pem on" `date`
echo --------------------------------------------------------------------
mkdir -p data/test.pem
local/prepare_acoustic_training_data.pl \
    --fragmentMarkers \-\*\~ \
    $test_data_dir data/test.pem > data/test.pem/skipped_utts.log || exit 1


echo -------------------------------------------------------------------
echo "Preparing test stm files in data/test.pem on" `date`
echo -------------------------------------------------------------------
local/prepare_stm.pl --fragmentMarkers \-\*\~ data/test.pem || exit 1
cp $glmFile data/test.pem/glm

steps/make_plp.sh \
    --cmd "$train_cmd" --nj $decode_nj \
    data/test.pem exp/make_plp/test.pem plp || exit 1
steps/compute_cmvn_stats.sh \
    data/test.pem exp/make_plp/test.pem plp || exit 1
# In case plp extraction failed on some utterance, delist them
utils/fix_data_dir.sh data/test.pem

##We prepare the subset -- no RTTM and no SUBSETTING
local/kws_setup.sh --case-insensitive $case_insensitive --rttm-file $rttm_file \
        --subset-ecf $test_data_list \
        $ecf_file $kwlist_file data/lang data/test.pem || exit 1

./local/create_shadow_dataset.sh data/shadow.pem data/dev data/test.pem

local/kws_data_prep.sh --case-insensitive $case_insensitive data/lang data/shadow.pem data/shadow.pem/kws

echo -------------------------------------------------------------------
echo "Preparing shadow stm files in data/shadow.pem on" `date`
echo -------------------------------------------------------------------
#steps/make_plp.sh \
#    --cmd "$train_cmd" --nj $decode_nj \
#    data/shadow.pem exp/make_plp/shadow.pem plp || exit 1
#steps/compute_cmvn_stats.sh \
#    data/shadow.pem exp/make_plp/shadow.pem plp || exit 1
# In case plp extraction failed on some utterance, delist them
utils/fix_data_dir.sh data/shadow.pem


if [[ -d exp/tri5 ]] ; then 

    echo --------------------------------------------------------------------------
    echo "Starting shadow.pem decoding using exp/tri4 on" `date`
    echo --------------------------------------------------------------------------
    mkdir -p exp/tri4/decode_shadow.pem
    steps/decode.sh --nj $decode_nj --cmd "$decode_cmd" \
        exp/tri4/graph data/shadow.pem exp/tri4/decode_shadow.pem &> exp/tri4/decode_shadow.pem.log

    echo --------------------------------------------------------------------------
    echo "Starting shadow.pem keyword spotting using exp/tri4 on" `date`
    echo --------------------------------------------------------------------------
    local/kws_search.sh --cmd "$decode_cmd" --duptime $duptime \
        data/lang data/shadow.pem exp/tri4/decode_shadow.pem &> exp/tri4/kws_shadow.pem.log

    echo --------------------------------------------------------------------------
    echo "Starting shadow.pem decoding using exp/tri5 on" `date`
    echo --------------------------------------------------------------------------
    mkdir -p exp/tri5/decode_shadow.pem
    steps/decode_fmllr.sh --nj $decode_nj --cmd "$decode_cmd" \
        exp/tri5/graph data/shadow.pem exp/tri5/decode_shadow.pem &> exp/tri5/decode_shadow.pem.log 

    echo --------------------------------------------------------------------------
    echo "Starting shadow.pem keyword spotting using exp/tri5 on" `date`
    echo --------------------------------------------------------------------------
    local/kws_search.sh --cmd "$decode_cmd" --duptime $duptime \
        data/lang data/shadow.pem exp/tri5/decode_shadow.pem &> exp/tri5/kws_shadow.pem.log
    local/kws_search.sh --cmd "$decode_cmd" --duptime $duptime \
        data/lang data/shadow.pem exp/tri5/decode_shadow.pem.si &> exp/tri5/kws_eval.si.pem.log
    
    echo --------------------------------------------------------------------------
    echo "Starting shadow.pem decoding using exp/sgmm5 on" `date`
    echo --------------------------------------------------------------------------
    steps/decode_sgmm2.sh \
        --nj $decode_nj --cmd "$decode_cmd" --transform-dir exp/tri5/decode_shadow.pem \
        exp/sgmm5/graph data/shadow.pem exp/sgmm5/decode_shadow.pem &> exp/sgmm5/decode_shadow.pem.log
    steps/decode_sgmm2.sh --use-fmllr true --nj $decode_nj --cmd "$decode_cmd" \
        --transform-dir exp/tri5/decode_shadow.pem \
        exp/sgmm5/graph data/shadow.pem exp/sgmm5/decode_fmllr_shadow.pem &> exp/sgmm5/decode_fmllr_shadow.pem.log

    echo --------------------------------------------------------------------------
    echo "Starting shadow.pem keyword spotting using exp/sgmm5 on" `date`
    echo --------------------------------------------------------------------------
    local/kws_search.sh --cmd "$decode_cmd" --duptime $duptime \
        data/lang data/shadow.pem exp/sgmm5/decode_shadow.pem &> exp/sgmm5/kws_shadow.pem.log
    local/kws_search.sh --cmd "$decode_cmd" --duptime $duptime \
        data/lang data/shadow.pem exp/sgmm5/decode_fmllr_shadow.pem &> exp/sgmm5/kws_shadow.pem.fmllr.log
    
    echo --------------------------------------------------------------------------
    echo "Starting exp/sgmm5_mmi_b0.1/decode[_fmllr] and keyword spotting on" `date`
    echo --------------------------------------------------------------------------
    for iter in 1 2 3 4; do
        steps/decode_sgmm2_rescore.sh \
            --cmd "$decode_cmd" --iter $iter --transform-dir exp/tri5/decode_shadow.pem \
            data/lang data/shadow.pem exp/sgmm5/decode_shadow.pem exp/sgmm5_mmi_b0.1/decode_shadow.pem_it$iter
        steps/decode_sgmm2_rescore.sh \
            --cmd "$decode_cmd" --iter $iter --transform-dir exp/tri5/decode_shadow.pem \
            data/lang data/shadow.pem exp/sgmm5/decode_fmllr_shadow.pem exp/sgmm5_mmi_b0.1/decode_fmllr_shadow.pem_it$iter
        
        local/kws_search.sh --cmd "$decode_cmd" --duptime $duptime \
            data/lang data/shadow.pem exp/sgmm5_mmi_b0.1/decode_shadow.pem_it$iter &> exp/sgmm5_mmi_b0.1/kws_shadow.pem.log
        local/kws_search.sh --cmd "$decode_cmd" --duptime $duptime \
            data/lang data/shadow.pem exp/sgmm5_mmi_b0.1/decode_fmllr_shadow.pem_it$iter &> exp/sgmm5_mmi_b0.1/kws_fmllr_shadow.pem.log
    done
else
    
    echo --------------------------------------------------------------------------
    echo "Starting FMLLR shadow.pem decoding using exp/tri4 on" `date`
    echo --------------------------------------------------------------------------
    mkdir -p exp/tri4/decode_shadow.pem
    steps/decode_fmllr.sh --nj $decode_nj --cmd "$decode_cmd" \
        exp/tri4/graph data/shadow.pem exp/tri4/decode_shadow.pem &> exp/tri4/decode_shadow.pem.log 

    echo --------------------------------------------------------------------------
    echo "Starting shadow.pem keyword spotting using exp/tri4 on" `date`
    echo --------------------------------------------------------------------------
    local/kws_search.sh --cmd "$decode_cmd" --duptime $duptime \
        data/lang data/shadow.pem exp/tri4/decode_shadow.pem &> exp/tri4/kws_shadow.pem.log
    local/kws_search.sh --cmd "$decode_cmd" --duptime $duptime \
        data/lang data/shadow.pem exp/tri4/decode_shadow.pem.si &> exp/tri4/kws_eval.si.pem.log

    echo --------------------------------------------------------------------------
    echo "Starting SGMM5 shadow.pem decoding using exp/sgmm5 on" `date`
    echo --------------------------------------------------------------------------
    steps/decode_sgmm2.sh \
        --nj $decode_nj --cmd "$decode_cmd" --transform-dir exp/tri4/decode_shadow.pem \
        exp/sgmm5/graph data/shadow.pem exp/sgmm5/decode_shadow.pem &> exp/sgmm5/decode_shadow.pem.log
    steps/decode_sgmm2.sh --use-fmllr true --nj $decode_nj --cmd "$decode_cmd" \
        --transform-dir exp/tri4/decode_shadow.pem \
        exp/sgmm5/graph data/shadow.pem exp/sgmm5/decode_fmllr_shadow.pem &> exp/sgmm5/decode_fmllr_shadow.pem.log
    
    echo --------------------------------------------------------------------------
    echo "Starting shadow.pem keyword spotting using exp/sgmm5 on" `date`
    echo --------------------------------------------------------------------------
    local/kws_search.sh --cmd "$decode_cmd" --duptime $duptime \
        data/lang data/shadow.pem exp/sgmm5/decode_shadow.pem &> exp/sgmm5/kws_shadow.pem.log
    local/kws_search.sh --cmd "$decode_cmd" --duptime $duptime \
        data/lang data/shadow.pem exp/sgmm5/decode_fmllr_shadow.pem &> exp/sgmm5/kws_shadow.pem.fmllr.log
    
    echo --------------------------------------------------------------------------
    echo "Starting exp/sgmm5_mmi_b0.1/decode[_fmllr] on" `date`
    echo --------------------------------------------------------------------------
    for iter in 1 2 3 4; do
        steps/decode_sgmm2_rescore.sh \
            --cmd "$decode_cmd" --iter $iter --transform-dir exp/tri4/decode_shadow.pem \
            data/lang data/shadow.pem exp/sgmm5/decode_shadow.pem exp/sgmm5_mmi_b0.1/decode_shadow.pem_it$iter
        steps/decode_sgmm2_rescore.sh \
            --cmd "$decode_cmd" --iter $iter --transform-dir exp/tri4/decode_shadow.pem \
            data/lang data/shadow.pem exp/sgmm5/decode_fmllr_shadow.pem exp/sgmm5_mmi_b0.1/decode_fmllr_shadow.pem_it$iter
        
        local/kws_search.sh --cmd "$decode_cmd" --duptime $duptime \
            data/lang data/shadow.pem exp/sgmm5_mmi_b0.1/decode_shadow.pem_it$iter &> exp/sgmm5_mmi_b0.1/kws_shadow.pem.log
        local/kws_search.sh --cmd "$decode_cmd" --duptime $duptime \
            data/lang data/shadow.pem exp/sgmm5_mmi_b0.1/decode_fmllr_shadow.pem_it$iter &> exp/sgmm5_mmi_b0.1/kws_fmllr_shadow.pem.log
    done
fi



echo -----------------------------------------------------
echo "Finished successfully on" `date`
echo -----------------------------------------------------

exit 0

