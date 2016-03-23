IFS=$'\n\t'
exp_dirs=(
#  chain
#  ctc
  make_dbl3
  make_hires
  make_hires_dbl
  make_hires_dbl2
  make_hiresf
  make_mfcc
  mono
  mono_ali
#  nnet3
  tri1
  tri1_ali
  tri2
  tri2_ali_100k_nodup
  tri2_ali_nodup
  tri3
  tri3_ali_nodup
  tri4
  tri4_ali_nodup
  tri4_ali_nodup_sp
  tri4_lats_nodup
  tri4_lats_nodup_sp
  tri4_lats_train_nodup_sp_ml8_max1
)

orig_dir=/home/dpovey/kaldi-chain/egs/swbd/s5c/exp/

for dir in ${exp_dirs[@]}; do
  ln -s $orig_dir/$dir exp/$dir
done