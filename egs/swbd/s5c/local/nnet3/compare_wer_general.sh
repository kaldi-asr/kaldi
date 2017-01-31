#!/bin/bash


# this script is used for comparing decoding results between systems.
# e.g. local/nnet3/compare_wer_general.sh tdnn_c_sp tdnn_d_sp
# For use with discriminatively trained systems you specify the epochs after a colon:
# for instance,
# local/nnet3/compare_wer_general.sh tdnn_d_sp tdnn_d_sp_smbr:1 tdnn_d_sp_smbr:2 ...

echo "# $0 $*";  # print command line.


echo -n "# System               "
for x in $*; do   printf " % 9s" $x;   done
echo


used_epochs=false

# this function set_names is used to separate the epoch-related parts of the name
# [for discriminative training] and the regular parts of the name.
# If called with a colon-free name, like:
#  set_names tdnn_a_sp
# it will set dir=exp/nnet3/tdnn_a_sp and epoch_suffix=""
# If called with something like:
#  set_names tdnn_d_sp_smbr:3
# it will set dir=exp/nnet3/tdnn_d_sp_smbr and epoch_suffix="epoch3"
set_names() {
  if [ $# != 1 ]; then
    echo "compare_wer_general.sh: internal error"
    exit 1  # exit the program
  fi
  name=$(echo $1 | cut -d: -f1)
  epoch=$(echo $1 | cut -s -d: -f2)
  dirname=exp/nnet3/$name
  if [ -z $epoch ]; then
    epoch_suffix=""
  else
    used_epochs=true
    epoch_suffix=_epoch${epoch}
  fi
}


echo -n "# WER on train_dev(tg) "
for x in $*; do
  set_names $x
  # note: the '*' in the directory name is because there
  # is _hires_ in there for the cross-entropy systems, and
  # nothing for the sequence trained systems.
  wer=$(grep WER $dirname/decode_train_dev*sw1_tg$epoch_suffix/wer_* | utils/best_wer.sh | awk '{print $2}')
  printf "% 10s" $wer
done
echo

echo -n "# WER on train_dev(fg) "
for x in $*; do
  set_names $x
  wer=$(grep WER $dirname/decode_train_dev*sw1_fsh_fg$epoch_suffix/wer_* | utils/best_wer.sh | awk '{print $2}')
  printf "% 10s" $wer
done
echo

echo -n "# WER on eval2000(tg)  "
for x in $*; do
  set_names $x
  wer=$(grep Sum $dirname/decode_eval2000*sw1_tg$epoch_suffix/score*/*ys | grep -v swbd | utils/best_wer.sh | awk '{print $2}')
  printf "% 10s" $wer
done
echo

echo -n "# WER on eval2000(fg)  "
for x in $*; do
  set_names $x
  wer=$(grep Sum $dirname/decode_eval2000*sw1_fsh_fg$epoch_suffix/score*/*ys | grep -v swbd | utils/best_wer.sh | awk '{print $2}')
  printf "% 10s" $wer
done
echo

if $used_epochs; then
  # we don't print the probs in this case.
  exit 0
fi

echo -n "# Final train prob     "
for x in $*; do
  set_names $x
  prob=$(grep log-likelihood $dirname/log/compute_prob_train.combined.log | awk '{print $8}')
  printf "% 10.3f" $prob
done
echo

echo -n "# Final valid prob     "
for x in $*; do
  set_names $x
  prob=$(grep log-likelihood $dirname/log/compute_prob_valid.combined.log | awk '{print $8}')
  printf "% 10.3f" $prob
done
echo
