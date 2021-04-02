#!/usr/bin/env bash


. conf/common_vars.sh
. ./lang.conf
. ./cmd.sh

set -e
set -o pipefail
set -u

function best_system_path_kws {
  path_to_outputs=$1

  best_out=`(find $path_to_outputs -name "sum.txt"  | xargs grep "^| *Occ")  | cut -f 1,13,17 -d '|' | sed 's/|//g'  |  sort -r -n -k 3 | head -n 1| awk '{print $1}'`
  echo `dirname $best_out`
}

function best_system_path_stt {
  path_to_outputs=$1
  best_out=` (find $path_to_outputs -name *.ctm.sys | xargs grep Avg)  | sed 's/|//g' | column -t | sort -n -k 9 | head -n 1|  awk '{print $1}' `
  echo `dirname $best_out`
}
# Wait till the main run.sh gets to the stage where's it's 
# finished aligning the tri5 model.

function lm_offsets {
  min=999
  for dir in "$@" ; do  
    lmw=${dir##*score_}

    [ $lmw -le $min ] && min=$lmw
  done

  lat_offset_str=""
  for dir in "$@" ; do  
    latdir_dir=`dirname $dir`
    lmw=${dir##*score_}
  
    offset=$(( $lmw - $min ))
    if [ $offset -gt 0 ] ; then
      lat_offset_str="$lat_offset_str ${latdir_dir}:$offset "
    else
      lat_offset_str="$lat_offset_str ${latdir_dir} "
    fi
  done

  echo $lat_offset_str

}

plp_kws=`best_system_path_kws "exp/sgmm5_mmi_b0.1/decode_fmllr_dev10h_it*/kws_*"`
plp_stt=`best_system_path_stt "exp/sgmm5_mmi_b0.1/decode_fmllr_dev10h_it*"`

dnn_kws=`best_system_path_kws "exp/tri6_nnet//decode_dev10h/kws_*"`
dnn_stt=`best_system_path_stt "exp/tri6_nnet/decode_dev10h/"`

bnf_kws=`best_system_path_kws "exp_bnf/sgmm7_mmi_b0.1/decode_fmllr_dev10h_it*/kws_*"`
bnf_stt=`best_system_path_stt "exp_bnf/sgmm7_mmi_b0.1/decode_fmllr_dev10h_it*"`



echo local/score_combine.sh --cmd "$decode_cmd" data/dev10h data/lang `lm_offsets $plp_stt $dnn_stt $bnf_stt` exp/combine/dev10h
#local/score_combine.sh --cmd "$decode_cmd" data/dev10h data/lang `lm_offsets $plp_stt $dnn_stt $bnf_stt` exp/combine/dev10h

echo local/kws_combine.sh --cmd "$decode_cmd" data/dev10h data/lang $plp_kws $dnn_kws $bnf_kws 
#local/kws_combine.sh --cmd "$decode_cmd" data/dev10h data/lang $plp_kws/kwslist.xml $dnn_kws/kwslist.xml $bnf_kws/kwslist.xml  exp/combine/dev10h/

mkdir -p exp/combine/kws_rescore
#local/rescoring/rescore_repeats.sh --cmd "$decode_cmd" \
#       exp/combine/dev10h/ data/dev10h data/train/text exp/combine/kws_rescore

exit 0
