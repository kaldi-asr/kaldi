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
  featdir=`pwd`/mfcc/nnet4b; mkdir -p $featdir
  mkdir -p conf/nnet4b
  all_fbankdirs=""
  all_mfccdirs=""
  pairs="1.1-1.0 1.05-1.2 1.0-0.8 0.95-1.1 0.9-0.9"
  for pair in $pairs; do
    vtln_warp=`echo $pair | cut -d- -f1`
    time_warp=`echo $pair | cut -d- -f2`
    fs=`perl -e "print ($time_warp*10);"`
    fbank_conf=conf/nnet4b/fbank_vtln${vtln_warp}_time${time_warp}.conf
    ( echo "--num-mel-bins=40"; echo "--frame-shift=$fs"; echo "--vtln-warp=$vtln_warp" ) > $fbank_conf
    echo "Making filterbank features for $pair"
    fbank_data=data/nnet4b/train_fbank_vtln${vtln_warp}_time${time_warp}
    all_fbankdirs="$all_fbankdirs $fbank_data"
    utils/copy_data_dir.sh --spk-prefix ${pair}- --utt-prefix ${pair}- data/train $fbank_data
    steps/make_fbank.sh --fbank-config $fbank_conf --nj 8 --cmd "run.pl" $fbank_data exp/nnet4b/make_mfcc/mfcc_$pair $featdir
    steps/compute_cmvn_stats.sh $fbank_data exp/nnet4b/fbank_$pair $featdir

    echo "Making MFCC features for $pair"      
    mfcc_data=data/nnet4b/train_mfcc_vtln${vtln_warp}_time${time_warp}
    mfcc_conf=conf/nnet4b/mfcc_vtln${vtln_warp}_time${time_warp}.conf
    ( echo "--use-energy=false"; echo "--frame-shift=$fs" ; echo "--vtln-warp=$vtln_warp" ) > $mfcc_conf
    utils/copy_data_dir.sh --spk-prefix ${pair}- --utt-prefix ${pair}- data/train $mfcc_data
    steps/make_mfcc.sh --mfcc-config $mfcc_conf --nj 8 --cmd "run.pl" $mfcc_data exp/nnet4b/make_mfcc/mfcc_$pair $featdir
    steps/compute_cmvn_stats.sh $mfcc_data exp/nnet4b/mfcc_$pair $featdir
    all_mfccdirs="$all_mfccdirs $mfcc_data"
  done

  utils/combine_data.sh data/nnet4b/train_fbank_all $all_fbankdirs
  utils/combine_data.sh data/nnet4b/train_mfcc_all $all_mfccdirs

  steps/align_fmllr.sh --nj 8 --cmd "$train_cmd" \
    data/nnet4b/train_mfcc_all data/lang exp/tri3b exp/tri3b_ali_nnet4b

  # In the combined filterbank directory, create a file utt2uniq which maps
  # our extended utterance-ids to "unique utterances".  This enables the
  # script steps/nnet2/get_egs.sh to hold out data in a more proper way.
  cat data/nnet4b/train_fbank_all/utt2spk | awk '{print $1;}' | \
    perl -ane ' chop; $utt = $_; s/[-0-9\.]+-[-0-9\.]+-//; print "$utt $_\n"; ' \
     > data/nnet4b/train_fbank_all/utt2uniq

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
      data/nnet4b/train_fbank_all data/lang exp/tri3b_ali_nnet4b exp/nnet4b  || exit 1
fi


if [ $stage -le 2 ]; then
  # Create the testing data.
  featdir=`pwd`/mfcc
  mkdir -p $featdir
  fbank_conf=conf/fbank_40.conf
  echo "--num-mel-bins=40" > $fbank_conf
  for x in test_mar87 test_oct87 test_feb89 test_oct89 test_feb91 test_sep92 train; do
    cp -rT data/$x data/${x}_fbank
    rm -r ${x}_fbank/split* || true
    steps/make_fbank.sh --fbank-config "$fbank_conf" --nj 8 \
      --cmd "run.pl" data/${x}_fbank exp/make_fbank/$x $featdir  || exit 1;
    steps/compute_cmvn_stats.sh data/${x}_fbank exp/make_fbank/$x $featdir  || exit 1;
  done
  utils/combine_data.sh data/test_fbank data/test_{mar87,oct87,feb89,oct89,feb91,sep92}_fbank
  steps/compute_cmvn_stats.sh data/test_fbank exp/make_fbank/test $featdir  
fi

if [ $stage -le 3 ]; then
   steps/nnet2/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj 20 --feat-type raw \
     exp/tri3b/graph data/test_fbank exp/nnet4b/decode
   steps/nnet2/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj 20 --feat-type raw \
     exp/tri3b/graph_ug data/test_fbank exp/nnet4b/decode_ug
fi

