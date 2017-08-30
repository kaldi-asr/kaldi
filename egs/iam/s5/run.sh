#!/bin/bash

stage=0
nj=20
color=1
data_dir=data
exp_dir=exp
. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.
. utils/parse_options.sh  # e.g. this parses the --stage option if supplied.

if [ $stage -le 0 ]; then
  # data preparation
  local/prepare_data.sh --nj $nj --dir $data_dir
fi

mkdir -p $data_dir/{train,val_1,val_2,test}/data
if [ $stage -le 1 ]; then
  for f in train val_1 val_2 test; do
    local/make_feature_vect.py $data_dir/$f --scale-size 40 | \
      copy-feats --compress=true --compression-method=7 \
      ark:- ark,scp:$data_dir/$f/data/images.ark,$data_dir/$f/feats.scp || exit 1

    steps/compute_cmvn_stats.sh $data_dir/$f || exit 1;
  done

fi

numSilStates=4
numStates=8
num_states=${numStates}states_${numSilStates}sil
num_gauss=10000

numLeavesTri=500
numGaussTri=20000

numLeavesMLLT=500
numGaussMLLT=20000
numLeavesSAT=500
numGaussSAT=20000

boost_sil=1
variance_floor_val=0.001
beam=20

lang_other=${num_states}
mono_other=OOV_pad_${lang_other}_${num_gauss}_var${variance_floor_val}_beam${beam}_boost${boost_sil}
tri_other=pad_${mono_other}_${numLeavesTri}_${numGaussTri}
tri2_other=pad_${tri_other}_${numLeavesMLLT}_${numGaussMLLT}
tri3_other=pad_${tri2_other}_${numLeavesSAT}_${numGaussSAT}

if [ $stage -le 2 ]; then
  local/prepare_dict.sh $data_dir/train/ $data_dir/test/ $data_dir/train/dict
  utils/prepare_lang.sh --num-sil-states $numSilStates --num-nonsil-states $numStates --position-dependent-phones false \
    $data_dir/train/dict "<sil>" $data_dir/lang_${lang_other}/temp $data_dir/lang_${lang_other}
fi

if [ $stage -le 3 ]; then
  cp -R $data_dir/lang_${lang_other} -T $data_dir/lang_test_${lang_other}

  local/add_test_data_to_train.py $data_dir/train/ $data_dir/test/ $data_dir/train/
  local/prepare_lm.sh --grammar-words false $data_dir/train/train_test $data_dir/lang_test_${lang_other} --ngram 2 || exit 1;
fi

if [ $stage -le 4 ]; then
  ## Starting basic training on features
  ## passing value for variance floor
  steps/train_mono.sh --nj $nj --variance_floor_val $variance_floor_val \
    --boost-silence $boost_sil \
    $data_dir/train \
    $data_dir/lang_${lang_other} \
    $exp_dir/mono_${mono_other}
fi

if [ $stage -le 5 ]; then
  utils/mkgraph.sh --mono $data_dir/lang_test_${lang_other} \
    $exp_dir/mono_${mono_other} \
    $exp_dir/mono_${mono_other}/graph
  steps/decode.sh --nj $nj --cmd $cmd \
    $exp_dir/mono_${mono_other}/graph \
    $data_dir/test \
    $exp_dir/mono_${mono_other}/decode_test
fi

if [ $stage -le 6 ]; then
  steps/align_si.sh --nj $nj \
    $data_dir/train $data_dir/lang_${lang_other} \
    $exp_dir/mono_${mono_other} \
    $exp_dir/mono_ali_${mono_other}
  steps/train_deltas.sh \
    --boost-silence $boost_sil \
    $numLeavesTri $numGaussTri $data_dir/train $data_dir/lang_${lang_other} \
    $exp_dir/mono_ali_${mono_other} \
    $exp_dir/tri_${tri_other}
fi

if [ $stage -le 7 ]; then
  utils/mkgraph.sh $data_dir/lang_test_${lang_other} \
    $exp_dir/tri_${tri_other} \
    $exp_dir/tri_${tri_other}/graph
  steps/decode.sh --nj $nj --cmd $cmd \
    $exp_dir/tri_${tri_other}/graph \
    $data_dir/test \
    $exp_dir/tri_${tri_other}/decode_test
fi

if [ $stage -le 8 ]; then
  steps/align_si.sh --nj $nj --cmd $cmd \
    $data_dir/train $data_dir/lang_${lang_other} \
    $exp_dir/mono_${mono_other} \
    $exp_dir/mono_ali_${mono_other}
  steps/train_lda_mllt.sh --cmd $cmd \
    --splice-opts "--left-context=3 --right-context=3" \
    $numLeavesMLLT $numGaussMLLT \
    $data_dir/train $data_dir/lang_${lang_other} \
    $exp_dir/mono_ali_${mono_other} $exp_dir/tri2_${tri2_other}
fi

if [ $stage -le 0 ]; then
  utils/mkgraph.sh $data_dir/lang_test_${lang_other} \
    $exp_dir/tri2_${tri2_other} \
    $exp_dir/tri2_${tri2_other}/graph
  steps/decode.sh --nj $nj --cmd $cmd \
    $exp_dir/tri2_${tri2_other}/graph \
    $data_dir/test \
    $exp_dir/tri2_${tri2_other}/decode_test
fi

if [ $stage -le 10 ]; then
  steps/align_fmllr.sh --nj $nj --cmd $cmd \
    --use-graphs true \
    $data_dir/train $data_dir/lang_${lang_other} \
    $exp_dir/tri2_${tri2_other} \
    $exp_dir/tri2_ali_${tri2_other}
  steps/train_sat.sh --cmd $cmd \
    $numLeavesSAT $numGaussSAT \
    $data_dir/train $data_dir/lang_${lang_other} \
    $exp_dir/tri2_ali_${tri2_other} $exp_dir/tri3_${tri3_other}
fi

if [ $stage -le 11 ]; then
  utils/mkgraph.sh $data_dir/lang_test_${lang_other} \
    $exp_dir/tri3_${tri3_other} \
    $exp_dir/tri3_${tri3_other}/graph
  steps/decode_fmllr.sh --nj $nj --cmd $cmd \
    $exp_dir/tri3_${tri3_other}/graph \
    $data_dir/test \
    $exp_dir/tri3_${tri3_other}/decode_test
fi

if [ $stage -le 12 ]; then
  steps/align_si.sh --nj $nj --cmd $cmd \
    --use-graphs true \
    $data_dir/train $data_dir/lang_${lang_other} \
    $exp_dir/tri2_${tri2_other} \
    $exp_dir/tri2_ali_${tri2_other}
fi
