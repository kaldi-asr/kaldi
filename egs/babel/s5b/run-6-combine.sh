#!/usr/bin/env bash

source_sys=shadow.seg
master_sys=dev10h.seg

. conf/common_vars.sh
. ./lang.conf
. ./cmd.sh

set -e
set -o pipefail
set -u


#systems=""; for sys in `ls -1 release/*dev*.ctm | grep -v RESCORED` ; do 
#  q=`readlink -f $sys`; 
#  decode=${q%%/dev10h.seg*}; 
#  w=`dirname ${q##*score_}`; 
#  echo $w; 
#  systems+=" ${decode}:$(($w - 10))";  
#done ; 
#echo $systems; 
#local/score_combine.sh --max-lmwt 16 --skip-scoring true --parallel-opts "-pe smp 2" --cmd "$decode_cmd" data/shadow.seg data/lang $systems exp/4way_combo/shadow.seg


#-systems=""; for sys in `ls -1 release/*c-*dev*kwslist.xml | grep BaDev | grep -v unnorm | grep -v oov | grep -v eval`  ; do 
#-  q=`readlink -f $sys`;
#-  echo $sys " -> " $q
#-  decode=`dirname $q`; 
#-  w=`dirname ${q##*kws_}`; 
#-  echo $w; 
#-  #systems+=" ${decode}:$(($w - 10))";  
#-  systems+=" ${decode}";  
#-done ; 
#-echo $systems; 
#-local/kws_combine.sh --cmd "$decode_cmd" --skip-scoring true --extraid dev data/shadow.seg data/lang $systems exp/4way_combo/shadow.seg
#-
#-systems=""; for sys in `ls -1 release/*eval*.xml | grep BaDev | grep -v unnorm | grep -v oov`  ; do 
#-  q=`readlink -f $sys`;
#-  echo $q
#-  decode=`dirname $q`; 
#-  w=`dirname ${q##*kws_}`; 
#-  echo $w; 
#-  #systems+=" ${decode}:$(($w - 10))";  
#-  systems+=" ${decode}";  
#-done ; 
#-echo $systems; 
#-local/kws_combine.sh --cmd "$decode_cmd" --skip-scoring true --extraid eval data/shadow.seg data/lang $systems exp/4way_combo/shadow.seg

./local/nist_eval/filter_data.sh  --cmd "$decode_cmd"  data/shadow.seg dev10h.seg exp/4way_combo/shadow.seg

exit 0
