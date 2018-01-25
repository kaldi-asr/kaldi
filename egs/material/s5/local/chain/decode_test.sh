#!/bin/bash
export LD_LIBRARY_PATH=/home/dpovey/libs

# Set -e here so that we catch if any executable fails immediately
set -euo pipefail

language=swahili
datadev="data/"$language"/analysis1"
label_delay=5
tlstm_affix=1a   # affix for the TDNN-LSTM directory name
test_sets="analysis1-segmented"
nnet3_affix=
tree_dir=exp/$language/chain${nnet3_affix}/tree_sp${tree_affix:+_$tree_affix}             
dir=exp/$language/chain${nnet3_affix}/tdnn_lstm${tlstm_affix}_sp
if [ $label_delay -gt 0 ]; then dir=${dir}_ld$label_delay; fi
train_set=train
train_data_dir=data/$language/${train_set}_sp_hires                                       
lores_train_data_dir=data/$language/${train_set}_sp

decode_nj=30
gmm=tri3
nnet3_affix=

tlstm_affix=1a   # affix for the TDNN-LSTM directory name
tree_affix=

# training options
# training chunk-options
chunk_width=140,100,160
chunk_left_context=40
chunk_right_context=0
label_delay=5
common_egs_dir=
xent_regularize=0.1


# End configuration section.
echo "$0 $@"  # Print the command line for logging

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

if ! cuda-compiled; then
  cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi

# audio segmentation
local/preprocess_test.sh --datadev $datadev

nj=30
gmm=tri3
test_sets="analysis1-segmented"

# stage 3

for datadir in $test_sets; do
  utils/copy_data_dir.sh data/$language/$datadir data/$language/${datadir}_hires
done

for datadir in $test_sets; do
  steps/make_mfcc.sh --nj $nj --mfcc-config conf/mfcc_hires.conf \
    --cmd "$train_cmd" data/$language/${datadir}_hires || exit 1;
  steps/compute_cmvn_stats.sh data/$language/${datadir}_hires || exit 1;
  utils/fix_data_dir.sh data/$language/${datadir}_hires || exit 1;
done

# stage 6

# extract iVectors for the test data, in this case we don't need the speed
# perturbation (sp).
for data in $test_sets; do
  steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj $nj \
    data/$language/${data}_hires exp/$language/nnet3${nnet3_affix}/extractor \
    exp/$language/nnet3${nnet3_affix}/ivectors_${data}_hires
done


gmm_dir=exp/$language/$gmm                                                                
ali_dir=exp/$language/${gmm}_ali_${train_set}_sp                                          
tree_dir=exp/$language/chain${nnet3_affix}/tree_sp${tree_affix:+_$tree_affix}             
dir=exp/$language/chain${nnet3_affix}/tdnn_lstm${tlstm_affix}_sp                          
if [ $label_delay -gt 0 ]; then dir=${dir}_ld$label_delay; fi                   
train_data_dir=data/$language/${train_set}_sp_hires                                       
lores_train_data_dir=data/$language/${train_set}_sp                                       
train_ivector_dir=exp/$language/nnet3${nnet3_affix}/ivectors_${train_set}_sp_hires        
                                                                                
for f in $gmm_dir/final.mdl $train_data_dir/feats.scp $train_ivector_dir/ivector_online.scp \
    $lores_train_data_dir/feats.scp $ali_dir/ali.1.gz; do                       
  [ ! -f $f ] && echo "$0: expected file $f to exist" && exit 1                 
done 

# stage 16

frames_per_chunk=$(echo $chunk_width | cut -d, -f1)
rm $dir/.error 2>/dev/null || true

for data in $test_sets; do
  (
    nspk=$(wc -l <data/$language/${data}_hires/spk2utt)
    steps/nnet3/decode.sh \
        --acwt 1.0 --post-decode-acwt 10.0 \
        --extra-left-context $chunk_left_context \
        --extra-right-context $chunk_right_context \
        --extra-left-context-initial 0 \
        --extra-right-context-final 0 \
        --frames-per-chunk $frames_per_chunk \
        --nj $nspk --cmd "$decode_cmd"  --num-threads 4 \
        --online-ivector-dir exp/$language/nnet3${nnet3_affix}/ivectors_${data}_hires \
        $tree_dir/graph data/$language/${data}_hires ${dir}/decode_${data} || exit 1
  ) || touch $dir/.error &
done
wait
[ -f $dir/.error ] && echo "$0: there was a problem while decoding" && exit 1

# resolve ctm overlaping regions, and compute wer
local/postprocess_test.sh --test_sets $test_sets --tree_dir $tree_dir \
  --dir $dir --language $language

exit 0;
