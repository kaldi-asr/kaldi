
# this script is used for comparing decoding results between systems.
# e.g. local/chain/compare_wer_general.sh tdnn_c_sp tdnn_d_sp
# For use with discriminatively trained systems you specify the epochs after a colon:
# for instance,
# local/chain/compare_wer_general.sh tdnn_d_sp tdnn_d_sp_smbr:1 tdnn_d_sp_smbr:2 ...

echo "# $0 $*";  # print command line.

include_looped=false
if [ "$1" == "--looped" ]; then
  include_looped=true
  shift
fi

echo -n "# System               "
for x in $*; do   printf " % 9s" $x;   done
echo


used_epochs=false

# this function set_names is used to separate the epoch-related parts of the name
# [for discriminative training] and the regular parts of the name.
# If called with a colon-free name, like:
#  set_names tdnn_a
# it will set dir=exp/chain/tdnn_a and epoch_suffix=""
# If called with something like:
#  set_names tdnn_d_smbr:3
# it will set dir=exp/chain/tdnn_d_smbr and epoch_suffix="epoch3"
set_names() {
  if [ $# != 1 ]; then
    echo "compare_wer_general.sh: internal error"
    exit 1  # exit the program
  fi
  name=$(echo $1 | cut -d: -f1)
  epoch=$(echo $1 | cut -s -d: -f2)
  dirname=exp/chain/$name
  if [ -z $epoch ]; then
    epoch_suffix=""
  else
    used_epochs=true
    epoch_suffix=_epoch${epoch}
  fi
}

echo -n "# WER on eval2000(tg)  "
for x in $*; do
  set_names $x
  wer=$(grep Sum $dirname/decode_eval2000_fsh_sw1_tg$epoch_suffix/score*/eval2000_hires.ctm.filt.sys | grep -v swbd | utils/best_wer.sh | awk '{print $2}')
  printf "% 10s" $wer
done
echo

if $include_looped; then
  echo -n "#           [looped:]  "
  for x in $*; do
    set_names $x
    wer=$(grep Sum $dirname/decode_eval2000_fsh_sw1_tg${epoch_suffix}_looped/score*/eval2000_hires.ctm.filt.sys | grep -v swbd | utils/best_wer.sh | awk '{print $2}')
    printf "% 10s" $wer
  done
  echo
fi

echo -n "# WER on eval2000(fg)  "
for x in $*; do
  set_names $x
  wer=$(grep Sum $dirname/decode_eval2000_fsh_sw1_fg$epoch_suffix/score*/eval2000_hires.ctm.filt.sys | grep -v swbd | utils/best_wer.sh | awk '{print $2}')
  printf "% 10s" $wer
done
echo

if $include_looped; then
  echo -n "#           [looped:]  "
  for x in $*; do
    set_names $x
    wer=$(grep Sum $dirname/decode_eval2000_fsh_sw1_fg${epoch_suffix}_looped/score*/eval2000_hires.ctm.filt.sys | grep -v swbd | utils/best_wer.sh | awk '{print $2}')
    printf "% 10s" $wer
  done
  echo
fi

echo -n "# WER on rt03(tg)  "
for x in $*; do
  set_names $x
  wer=$(grep Sum $dirname/decode_rt03_fsh_sw1_tg$epoch_suffix/score*/rt03_hires.ctm.filt.sys| grep -v swbd | utils/best_wer.sh | awk '{print $2}')
  printf "% 10s" $wer
done
echo

if $include_looped; then
  echo -n "#           [looped:]  "
  for x in $*; do
    set_names $x
    wer=$(grep Sum $dirname/decode_rt03_fsh_sw1_tg${epoch_suffix}_looped/score*/rt03_hires.ctm.filt.sys| grep -v swbd | utils/best_wer.sh | awk '{print $2}')
    printf "% 10s" $wer
  done
  echo
fi

echo -n "# WER on rt03(fg)  "
for x in $*; do
  set_names $x
  wer=$(grep Sum $dirname/decode_rt03_fsh_sw1_fg$epoch_suffix/score*/rt03_hires.ctm.filt.sys| grep -v swbd | utils/best_wer.sh | awk '{print $2}')
  printf "% 10s" $wer
done
echo

if $include_looped; then
  echo -n "#           [looped:]  "
  for x in $*; do
    set_names $x
    wer=$(grep Sum $dirname/decode_rt03_fsh_sw1_fg${epoch_suffix}_looped/score*/rt03_hires.ctm.filt.sys | grep -v swbd | utils/best_wer.sh | awk '{print $2}')
    printf "% 10s" $wer
  done
  echo
fi

if $used_epochs; then
  # we don't print the probs in this case.
  exit 0
fi


echo -n "# Final train prob     "
for x in $*; do
  prob=$(grep Overall exp/chain/${x}/log/compute_prob_train.final.log | grep -v xent | awk '{print $8}')
  printf "% 10.3f" $prob
done
echo

echo -n "# Final valid prob     "
for x in $*; do
  prob=$(grep Overall exp/chain/${x}/log/compute_prob_valid.final.log | grep -v xent | awk '{print $8}')
  printf "% 10.3f" $prob
done
echo

echo -n "# Final train prob (xent)    "
for x in $*; do
  prob=$(grep Overall exp/chain/${x}/log/compute_prob_train.final.log | grep -w xent | awk '{print $8}')
  printf "% 10.3f" $prob
done
echo

echo -n "# Final valid prob (xent)    "
for x in $*; do
  prob=$(grep Overall exp/chain/${x}/log/compute_prob_valid.final.log | grep -w xent | awk '{print $8}')
  printf "% 10.4f" $prob
done
echo
