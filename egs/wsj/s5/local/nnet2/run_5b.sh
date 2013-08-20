#!/bin/bash


stage=0
train_stage=-100
# This trains only unadapted (just cepstral mean normalized) features,
# and uses various combinations of VTLN warping factor and time-warping
# factor to artificially expand the amount of data.

. cmd.sh

. utils/parse_options.sh  # to parse the --stage option, if given

[ $# != 0 ] && echo "Usage: local/run_4b.sh [--stage <stage> --train-stage <train-stage>]" && exit 1;

set -e

if [ $stage -le 0 ]; then 
  # Create the training data.
  featdir=`pwd`/mfcc/nnet5b; mkdir -p $featdir
  fbank_conf=conf/fbank_40.conf
  echo "--num-mel-bins=40" > $fbank_conf
  steps/nnet2/get_perturbed_feats.sh --cmd "$train_cmd" \
    $fbank_conf $featdir exp/perturbed_fbanks_si284 data/train_si284 data/train_si284_perturbed_fbank &
  steps/nnet2/get_perturbed_feats.sh --cmd "$train_cmd" --feature-type mfcc \
    conf/mfcc.conf $featdir exp/perturbed_mfcc_si284 data/train_si284 data/train_si284_perturbed_mfcc &
  wait
fi

if [ $stage -le 1 ]; then
  steps/align_fmllr.sh --nj 30 --cmd "$train_cmd" \
    data/train_si284_perturbed_mfcc data/lang exp/tri4b exp/tri4b_ali_si284_perturbed_mfcc
fi 

if [ $stage -le 2 ]; then
  steps/nnet2/train_block.sh \
     --cleanup false \
     --initial-learning-rate 0.01 --final-learning-rate 0.001 \
     --num-epochs 10 --num-epochs-extra 5 \
     --cmd "$decode_cmd" \
     --hidden-layer-dim 1536 \
     --num-block-layers 3 --num-normal-layers 3 \
      data/train_si284_perturbed_fbank data/lang exp/tri4b_ali_si284_perturbed_mfcc exp/nnet5b  || exit 1
fi

if [ $stage -le 3 ]; then # create testing fbank data.
  featdir=`pwd`/mfcc
  fbank_conf=conf/fbank_40.conf
  for x in test_eval92 test_eval93 test_dev93; do 
    cp -rT data/$x data/${x}_fbank
    rm -r ${x}_fbank/split* || true
    steps/make_fbank.sh --fbank-config "$fbank_conf" --nj 8 \
      --cmd "$train_cmd" data/${x}_fbank exp/make_fbank/$x $featdir  || exit 1;
    steps/compute_cmvn_stats.sh data/${x}_fbank exp/make_fbank/$x $featdir  || exit 1;
  done
fi

if [ $stage -le 4 ]; then
  steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 10 \
     exp/tri4b/graph_bd_tgpr data/test_dev93_fbank exp/nnet5b/decode_bd_tgpr_dev93

  steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 8 \
     exp/tri4b/graph_bd_tgpr data/test_eval92_fbank exp/nnet5b/decode_bd_tgpr_eval92
fi



exit 0;

  mkdir -p conf/nnet5b
  all_fbankdirs=""
  all_mfccdirs=""
  pairs="1.1-1.0 1.05-1.2 1.0-0.8 0.95-1.1 0.9-0.9"
  for pair in $pairs; do
    vtln_warp=`echo $pair | cut -d- -f1`
    time_warp=`echo $pair | cut -d- -f2`
    fs=`perl -e "print ($time_warp*10);"`
    fbank_conf=conf/nnet5b/fbank_vtln${vtln_warp}_time${time_warp}.conf
    ( echo "--num-mel-bins=40"; echo "--frame-shift=$fs"; echo "--vtln-warp=$vtln_warp" ) > $fbank_conf
    echo "Making filterbank features for $pair"
    fbank_data=data/nnet5b/train_si284_fbank_vtln${vtln_warp}_time${time_warp}
    all_fbankdirs="$all_fbankdirs $fbank_data"
    utils/copy_data_dir.sh --spk-prefix ${pair}- --utt-prefix ${pair}- data/train_si284 $fbank_data
    steps/make_fbank.sh --fbank-config $fbank_conf --nj 8 --cmd "run.pl" $fbank_data exp/nnet5b/make_mfcc/mfcc_$pair $featdir
    steps/compute_cmvn_stats.sh $fbank_data exp/nnet5b/fbank_$pair $featdir

    echo "Making MFCC features for $pair"      
    mfcc_data=data/nnet5b/train_si284_mfcc_vtln${vtln_warp}_time${time_warp}
    mfcc_conf=conf/nnet5b/mfcc_vtln${vtln_warp}_time${time_warp}.conf
    ( echo "--use-energy=false"; echo "--frame-shift=$fs" ; echo "--vtln-warp=$vtln_warp" ) > $mfcc_conf
    utils/copy_data_dir.sh --spk-prefix ${pair}- --utt-prefix ${pair}- data/train_si284 $mfcc_data
    steps/make_mfcc.sh --mfcc-config $mfcc_conf --nj 8 --cmd "run.pl" $mfcc_data exp/nnet5b/make_mfcc/mfcc_$pair $featdir
    steps/compute_cmvn_stats.sh $mfcc_data exp/nnet5b/mfcc_$pair $featdir
    all_mfccdirs="$all_mfccdirs $mfcc_data"
  done

  utils/combine_data.sh data/nnet5b/train_si284_fbank_all $all_fbankdirs
  utils/combine_data.sh data/nnet5b/train_si284_mfcc_all $all_mfccdirs

  steps/align_fmllr.sh --nj 8 --cmd "$train_cmd" \
    data/nnet5b/train_si284_mfcc_all data/lang exp/tri3b exp/tri3b_ali_nnet5b

  # In the combined filterbank directory, create a file utt2uniq which maps
  # our extended utterance-ids to "unique utterances".  This enables the
  # script steps/nnet2/get_egs.sh to hold out data in a more proper way.
  cat data/nnet5b/train_si284_fbank_all/utt2spk | awk '{print $1;}' | \
    perl -ane ' chop; $utt = $_; s/[-0-9\.]+-[-0-9\.]+-//; print "$utt $_\n"; ' \
     > data/nnet5b/train_si284_fbank_all/utt2uniq

fi

if [ $stage -le 1 ]; then
  steps/nnet2/train_block.sh --stage "$train_stage" \
     --bias-stddev 0.5 --splice-width 7 --egs-opts "--feat-type raw" \
     --softmax-learning-rate-factor 0.5 --cleanup false \
     --initial-learning-rate 0.04 --final-learning-rate 0.004 \
     --num-epochs-extra 10 --add-layers-period 1 \
     --mix-up 4000 \
     --cmd "$decode_cmd" \
     --hidden-layer-dim 450 \
      data/nnet5b/train_si284_fbank_all data/lang exp/tri3b_ali_nnet5b exp/nnet5b  || exit 1
fi


if [ $stage -le 2 ]; then
  # Create the testing data.
  featdir=`pwd`/mfcc
  mkdir -p $featdir
  fbank_conf=conf/fbank_40.conf
  echo "--num-mel-bins=40" > $fbank_conf
  for x in test_eval92 test_eval93 test_dev93; do
    cp -rT data/$x data/${x}_fbank
    rm -r ${x}_fbank/split* || true
    steps/make_fbank.sh --fbank-config "$fbank_conf" --nj 8 \
      --cmd "run.pl" data/${x}_fbank exp/make_fbank/$x $featdir  || exit 1;
    steps/compute_cmvn_stats.sh data/${x}_fbank exp/make_fbank/$x $featdir  || exit 1;
  done
fi

if [ $stage -le 3 ]; then
   steps/nnet2/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj 20 --feat-type raw \
     exp/tri3b/graph data/test_fbank exp/nnet5b/decode
   steps/nnet2/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj 20 --feat-type raw \
     exp/tri3b/graph_ug data/test_fbank exp/nnet5b/decode_ug
fi

