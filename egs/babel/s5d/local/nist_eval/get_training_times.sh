if [ -z $1 ] ; then
  dir=`pwd`
else
  dir=$1
fi
echo $dir


convertsecs() {
    h=$(($1/3600))
    m=$((($1/60)%60))
    s=$(($1%60))
    printf "%02d:%02d:%02d\n" $h $m $s
}

function process {
  count=1
  if [ ! -z $1 ];  then
    count=$1
  fi

  replace=""
  for a in `seq 1 $count` ; do
    replace+="\t"
  done

  ( 
    eval `grep "group=all"` 
    echo -n "threads=$total_threads"
    echo -n " cpu_time=$total_cpu_time wall_time=$clock_time"
    echo -n " human_cpu_time="`convertsecs $total_cpu_time`
    echo -n " human_wall_time="`convertsecs $clock_time`
    echo ""
  ) | sed 's/^/'$replace'/g'
}

function legend {
  echo -ne '"'"$@"'" '
}

legend Parameterization dev/train
local/summarize_logs.pl $dir/exp/make_*/*train*/  |  process

if [ -d $dir/data/local/extend ] ; then
  legend "Extending the lexicon"
  local/summarize_logs.pl $dir/data/local/extend/tmp/log | process 
fi

legend "Training upto stage tri5"
local/summarize_logs.pl $dir/exp/mono*/log $dir/exp/tri{1..5}/log $dir/exp/tri{1..4}_ali*/log | process 

legend "SGMM2 stage training"
local/summarize_logs.pl $dir/exp/ubm5/log $dir/exp/sgmm5/log $dir/exp/tri5_ali/log  | process 

legend "SGMM2+bMMI stage training"
local/summarize_logs.pl $dir/exp/sgmm5_*/log $dir/exp/ubm5/log $dir/exp/sgmm5_denlats/log/* | process 

nnet=tri6_nnet
[ ! -d $dir/exp/$nnet ] && nnet=tri6b_nnet

legend "DNN stage training GPU"
local/summarize_logs.pl $dir/exp/$nnet/log  | process

legend "BNF stage training"
local/summarize_logs.pl $dir/exp_bnf/tri6_bnf/log  | process

legend "BNF stage training GPU"
local/summarize_logs.pl $dir/exp_bnf/tri{5,6}/log $dir/exp_bnf/sgmm7*/log \
  $dir/exp_bnf/sgmm7_denlats/log/*  $dir/exp_bnf/ubm7 | process

legend "SEGMENTATION TRAINING: "
local/summarize_logs.pl $dir/exp/tri4_train_seg_ali/log \
  $dir/exp/make_plp_pitch/train_seg/ \
  $dir/exp/tri4b_seg/log | process

semisup=exp_bnf_semisup2
if [ -d $dir/param_bnf_semisup ] || [ -d $dir/param_bnf_semisup2 ]  ; then
  [ ! -d $dir/$semisup ] && semisup=exp_bnf_semisup

  decode=unsup.seg
  legend "BNF_SEMISUP training, segmentation "
  local/summarize_logs.pl $dir/exp/make_seg/$decode/log \
    $dir/exp/make_seg/$decode/make_plp/ \
    $dir/exp/tri4b_seg/decode_${decode}/log \
    $dir/exp/make_plp/$decode | process

  legend "BNF_SEMISUP training, ecode unsup.seg TRI5 "
  local/summarize_logs.pl $dir/exp/tri5/decode_*${decode}*/log | process
  legend "BNF_SEMISUP training, ecode unsup.seg PLP "
  local/summarize_logs.pl $dir/exp/{sgmm5,sgmm5_mmi_b0.1}/decode_*${decode}*/log | process
  legend "BNF_SEMISUP training, ecode unsup.seg DNN "
  local/summarize_logs.pl $dir/exp/$nnet/decode_*${decode}*/log | process
  legend "BNF_SEMISUP training, data preparation for BNF_SEMISUP "
  local/summarize_logs.pl $dir/exp/combine2_post/unsup.seg/log \
    $dir/exp/combine2_post/unsup.seg/decode_unsup.seg/log\
    $dir/exp/tri6_nnet_ali/log | process

  legend "BNF_SEMISUP training, TRAIN BNF_SEMISUP BNF GPU "
  local/summarize_logs.pl $dir/$semisup/tri6_bnf/log  | process
  legend "BNF_SEMISUP training, TRAIN BNF_SEMISUP BNF "
  local/summarize_logs.pl $dir/$semisup/tri{5,6}/log $dir/exp_bnf/sgmm7*/log \
    $dir/exp_bnf/sgmm7_denlats/log/* $dir/exp_bnf/ubm7 | process
fi

if [ -d $dir/exp/tri6_nnet_mpe ] ; then
  legend "DNN_MPE stage CPU training"
  local/summarize_logs.pl $dir/exp/tri6_nnet_ali/log/ \
    $dir/exp/tri6_nnet_denlats/log/* | process

  legend "DNN_MPE stage GPU training"
  local/summarize_logs.pl $dir/exp/tri6_nnet_mpe/log/ | process
fi

#~decode=dev10h.seg
#~legend "DEV10H.SEG decoding"
#~legend "Segmentation: "
#~local/summarize_logs.pl $dir/exp/make_seg/$decode/log \
#~     $dir/exp/make_seg/$decode/make_plp/ \
#~     $dir/exp/tri4b_seg/decode_${decode}/log \
#~     $dir/exp/make_plp/$decode | process
#~legend "Decode $decode TRI5: "
#~local/summarize_logs.pl $dir/exp/tri5/decode_*${decode}*/log | process
#~legend "Decode $decode PLP: "
#~local/summarize_logs.pl $dir/exp/{sgmm5,sgmm5_mmi_b0.1}/decode_*${decode}*/log | process
#~legend "Decode $decode DNN: "
#~local/summarize_logs.pl $dir/exp/$nnet/decode_*${decode}*/log | process
#~legend "Decode $decode PLP: "
#~local/summarize_logs.pl $dir/exp/{sgmm5,sgmm5_mmi_b0.1}/decode_*${decode}*/log | process

legend "G2P and confusion matrix: "
local/summarize_logs.pl  $dir/exp/conf_matrix/log  $dir/exp/g2p/log  | process
if [ -d $dir/data/shadow2.uem ]; then
  decode=shadow2.uem
else
  decode=shadow.uem
fi

legend "Segmentation $decode: provided..."
echo
#--legend "Segmentation: "
#--local/summarize_logs.pl $dir/exp/make_seg/$decode/log \
#--     $dir/exp/make_seg/$decode/make_plp/ \
#--     $dir/exp/tri4b_seg/decode_${decode}/log \
#--     $dir/exp/make_plp/$decode | process
legend "Parametrization: "
local/summarize_logs.pl $dir/exp/make_plp/$decode |  process
legend "Decode $decode TRI5: "
local/summarize_logs.pl $dir/exp/tri5/decode_*${decode}*/log | process
legend "Decode $decode PLP: "
local/summarize_logs.pl $dir/exp/{sgmm5,sgmm5_mmi_b0.1}/decode_*${decode}*/log | process
legend "Decode $decode DNN: "
local/summarize_logs.pl $dir/exp/$nnet/decode_*${decode}*/log | process
legend "Decode $decode BNF: "
local/summarize_logs.pl $dir/exp_bnf/{tri6,sgmm7,sgmm7_mmi_b0.1}/decode_*${decode}*/log | process
if [ -d $dir/$semisup ] ; then
  legend "Decode $decode BNF_SEMISUP: "
  local/summarize_logs.pl $dir/$semisup/{tri6,sgmm7,sgmm7_mmi_b0.1}/decode_*${decode}*/log | process
fi
if [ -d $dir/exp/tri6_nnet_mpe ] ; then
  legend "Decode $decode DNN_MPE: "
  local/summarize_logs.pl $dir/exp/tri6_nnet_mpe/decode_${decode}_epoch*/log | process
fi

legend "Indexing $decode PLP: "
local/summarize_logs.pl $dir/exp/sgmm5_mmi_b0.1/decode_*${decode}*/kws_indices*/log | process
legend "Indexing $decode DNN: "
local/summarize_logs.pl $dir/exp/$nnet/decode_*${decode}*/kws_indices*/log | process
legend "Indexing $decode BNF: "
local/summarize_logs.pl $dir/exp_bnf/sgmm7_mmi_b0.1/decode_*${decode}*/kws_indices*/log | process
if [ -d $dir/$semisup ] ; then
  legend "Indexing $decode BNF_SEMISUP: "
  local/summarize_logs.pl $dir/$semisup/sgmm7_mmi_b0.1/decode_*${decode}*/kws_indices*/log | process
fi
if [ -d $dir/exp/tri6_nnet_mpe ] ; then
  legend "Indexing $decode DNN_MPE: "
  local/summarize_logs.pl $dir/exp/tri6_nnet_mpe/decode_${decode}_epoch*/kws_indices*/log | process
fi

legend "Search $decode PLP: "
local/summarize_logs.pl $dir/exp/sgmm5_mmi_b0.1/decode_*${decode}*/evalKW_kws \
  $dir/exp/sgmm5_mmi_b0.1/decode_*${decode}*/evalKW_kws_*/log | process
legend "Search $decode DNN: "
local/summarize_logs.pl $dir/exp/$nnet/decode_*${decode}*/evalKW_kws \
  $dir/exp/$nnet/decode_*${decode}*/evalKW_kws_*/log | process
legend "Search $decode BNF: "
local/summarize_logs.pl $dir/exp_bnf/sgmm7_mmi_b0.1/decode_*${decode}*/evalKW_kws \
  $dir/exp_bnf/sgmm7_mmi_b0.1/decode_*${decode}*/evalKW_kws_*/log | process
if [ -d $dir/$semisup ] ; then
  legend "Search $decode BNF_SEMISUP: "
  local/summarize_logs.pl $dir/$semisup/sgmm7_mmi_b0.1/decode_*${decode}*/evalKW_kws/ \
    $dir/$semisup/sgmm7_mmi_b0.1/decode_*${decode}*/evalKW_kws*/log | process
fi
if [ -d $dir/exp/tri6_nnet_mpe ] ; then
  legend "Search $decode DNN_MPE: "
  local/summarize_logs.pl $dir/exp/tri6_nnet_mpe/decode_${decode}_epoch*/evalKW_kws \
    $dir/exp/tri6_nnet_mpe/decode_${decode}_epoch*/evalKW_kws*/log | process
fi

legend "Proxies generation: "
local/summarize_logs.pl $dir/data/$decode/evalKW_oov_kws/g2p/log \
  $dir/data/$decode/evalKW_oov_kws/tmp/split/log  | process
legend "Search $decode PLP: "
local/summarize_logs.pl $dir/exp/sgmm5_mmi_b0.1/decode_*${decode}*/evalKW_oov_kws \
  $dir/exp/sgmm5_mmi_b0.1/decode_*${decode}*/evalKW_oov_kws_*/log | process
legend "Search $decode DNN: "
local/summarize_logs.pl $dir/exp/$nnet/decode_*${decode}*/evalKW_oov_kws \
  $dir/exp/$nnet/decode_*${decode}*/evalKW_oov_kws_*/log | process
legend "Search $decode BNF: "
local/summarize_logs.pl $dir/exp_bnf/sgmm7_mmi_b0.1/decode_*${decode}*/evalKW_oov_kws \
  $dir/exp_bnf/sgmm7_mmi_b0.1/decode_*${decode}*/evalKW_oov_kws_*/log | process

if [ -d $dir/$semisup ] ; then
  legend "Search $decode BNF_SEMISUP: "
  local/summarize_logs.pl $dir/$semisup/sgmm7_mmi_b0.1/decode_*${decode}*/evalKW_oov_kws/ \
    $dir/$semisup/sgmm7_mmi_b0.1/decode_*${decode}*/evalKW_oov_kws*/log | process
fi


if [ -d $dir/exp/tri6_nnet_mpe ] ; then
  legend "Search $decode DNN_MPE: "
  local/summarize_logs.pl $dir/exp/tri6_nnet_mpe/decode_${decode}_epoch*/evalKW_oov_kws \
    $dir/exp/tri6_nnet_mpe/decode_${decode}_epoch*/evalKW_oov_kws*/log | process
fi






