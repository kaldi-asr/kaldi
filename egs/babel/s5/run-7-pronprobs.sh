#!/bin/bash 


# Working out pronunciation probabilities and testing with them.

set -e           #Exit on non-zero return code from any command
set -o pipefail  #Exit if any of the commands in the pipeline will 
                 #return non-zero return code
set -u           #Fail on an undefined variable

. conf/common_vars.sh || exit 1;
. ./lang.conf || exit 1;

if [ ! -f data/local_withprob/.done ]; then
  echo -------------------------------------------------------------------------------
  echo "Creating lexicon with probabilities, data/local_withprob/lexicon.txt on `date`"
  echo -------------------------------------------------------------------------------

  cp -rT data/local data/local_withprob
  steps/get_lexicon_probs.sh data/train data/lang exp/tri5 data/local/lexicon.txt \
    exp/tri5_lexprobs data/local_withprob/lexiconp.txt || exit 1;
  touch data/local_withprob/.done
fi

mkdir -p data/lang_withprob
if [[ ! -f data/lang_withprob/.done || data/lang_withprob/L.fst -ot data/local/lexiconp.txt ]]; then
  echo ---------------------------------------------------------------------
  echo "Creating L.fst etc in data/lang_withprob" `date`
  echo ---------------------------------------------------------------------
  utils/prepare_lang.sh --share-silence-phones true \
    data/local_withprob $oovSymbol data/local_withprob/tmp.lang data/lang_withprob

  cp data/lang/G.fst data/lang_withprob/G.fst
  cmp data/lang/words.txt data/lang_withprob/words.txt || exit 1;
  touch data/lang_withprob/.done
fi


type=dev10h

eval my_data_dir=\$${type}_data_dir
eval my_data_list=\$${type}_data_list
if [ "$type" == dev2h ] ; then
  my_nj=$decode_nj

  eval my_ecf_file=$ecf_file 
  eval my_subset_ecf=$subset_ecf 
  eval my_kwlist_file=$kwlist_file 
  eval my_rttm_file=$rttm_file
elif [ "$type" == dev10h ] ; then
  eval my_nj=\$${type}_nj

  eval my_ecf_file=$ecf_file 
  eval my_subset_ecf=false
  eval my_kwlist_file=$kwlist_file 
  eval my_rttm_file=$rttm_file
else 
  eval my_nj=\$${type}_nj

  eval my_ecf_file=\$${type}_ecf_file 
  eval my_subset_ecf=\$${type}_subset_ecf 
  eval my_kwlist_file=\$${type}_kwlist_file 
  eval my_rttm_file=\$${type}_rttm_file
fi


if [ ! -f exp/tri5/decode_withprob_${type}/.done ]; then
  echo ---------------------------------------------------------------------
  echo "Spawning decoding with SAT models  on" `date`
  echo ---------------------------------------------------------------------
  mkdir -p exp/tri5/graph
  utils/mkgraph.sh \
      data/lang_withprob exp/tri5 exp/tri5/graph_withprob |tee exp/tri5/mkgraph_withprob.log
  mkdir -p exp/tri5/decode_withprob_${type}

  steps/decode_fmllr.sh --nj $my_nj \
    --cmd "$decode_cmd" "${decode_extra_opts[@]}" \
    exp/tri5/graph_withprob data/${type} exp/tri5/decode_withprob_${type} |tee exp/tri5/decode_withprob_${type}.log 
  touch exp/tri5/decode_withprob_${type}/.done
fi

if [ ! -f exp/tri5/decode_withprob_${type}/kws/.done ]; then
  local/kws_search.sh --cmd "$decode_cmd" --duptime $duptime \
      data/lang_withprob data/${type} exp/tri5/decode_withprob_${type}
  local/kws_search.sh --cmd "$decode_cmd" --duptime $duptime \
      data/lang_withprob data/${type} exp/tri5/decode_withprob_${type}.si
  touch exp/tri5/decode_withprob_${type}/kws/.done 
fi


exit 0;

