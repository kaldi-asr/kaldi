#!/bin/bash

# this script is used for comparing decoding results between systems.
# e.g. local/chain/compare_wer_general.sh exp/chain_cleaned/tdnn_{c,d}_sp
# For use with discriminatively trained systems you specify the epochs after a colon:
# for instance,
# local/chain/compare_wer_general.sh exp/chain_cleaned/tdnn_c_sp exp/chain_cleaned/tdnn_c_sp_smbr:{1,2,3}


echo "# $0 $*"

include_looped=false

. utils/parse_options.sh

used_epochs=false

# this function set_names is used to separate the epoch-related parts of the name
# [for discriminative training] and the regular parts of the name.
# If called with a colon-free directory name, like:
#  set_names exp/chain_cleaned/tdnn_lstm1e_sp_bi_smbr
# it will set dir=exp/chain_cleaned/tdnn_lstm1e_sp_bi_smbr and suffix=""
# If called with something like:
#  set_names exp/chain_cleaned/tdnn_d_sp_smbr:3
# it will set dir=exp/chain_cleaned/tdnn_d_sp_smbr and suffix="_epoch3"


set_names() {
  if [ $# != 1 ]; then
    echo "compare_wer_general.sh: internal error"
    exit 1  # exit the program
  fi
  dirname=$(echo $1 | cut -d: -f1)
  suffix=$(echo $1 | cut -s -d: -f2)
  model_name=$(echo $1 | cut -s -d: -f3)
  if [ ! -z "$suffix" ] && [[ $suffix =~ *epoch* ]]; then
    used_epochs=true
  else
    used_epochs=false
  fi
  if [ -z "$model_name" ]; then
    model_name=$(basename $dirname)
  fi
}



printf "# System %14s" ""
for x in $*; do   
  set_names $x  # sets $dirname and $suffix
  printf "% 10s" " $model_name";   done
echo

strings=("WER on dev    "  "WER on test    ")

for n in 0 1; do
   printf "# %s % 6s" "${strings[$n]}" ""
   for x in $*; do
     set_names $x  # sets $dirname and $suffix
     decode_names=(dev${suffix} test${suffix})
     wer=$(grep WER $dirname/decode_${decode_names[$n]}/wer* | utils/best_wer.sh | awk '{print $2}')
     printf "% 10s" $wer
   done
   echo
   if $include_looped; then
     printf "# % 20s" "         [looped:]"
     for x in $*; do
       set_names $x  # sets $dirname and $suffix
       decode_names=(dev${suffix} test${suffix})
       wer=$(grep WER $dirname/decode_looped_${decode_names[$n]}/wer* | utils/best_wer.sh | awk '{print $2}')
       printf "% 10s" $wer
     done
     echo
   fi
done


if $used_epochs; then
  exit 0;  # the diagnostics aren't comparable between regular and discriminatively trained systems.
fi

printf "# % 20s" "Final train prob     "
for x in $*; do
  set_names $x  # sets $dirname and $suffix
  prob=$(grep Overall $dirname/log/compute_prob_train.final.log | grep -v xent | awk '{printf("%.4f ", $8)}')
  printf "% 10s" $prob
done
echo

printf "# % 20s" "Final valid prob     "
for x in $*; do
  set_names $x  # sets $dirname and $suffix
  prob=$(grep Overall $dirname/log/compute_prob_valid.final.log | grep -v xent | awk '{printf("%.4f ", $8)}')
  printf "% 10s" $prob
done
echo

printf "# % 20s" "Final train prob (xent)"
for x in $*; do
  set_names $x  # sets $dirname and $suffix
  prob=$(grep Overall $dirname/log/compute_prob_train.final.log | grep -w xent | awk '{printf("%.4f ", $8)}')
  printf "% 10s" $prob
done
echo

printf "# % 20s" "Final valid prob (xent)"
for x in $*; do
  set_names $x  # sets $dirname and $suffix
  prob=$(grep Overall $dirname/log/compute_prob_valid.final.log | grep -w xent | awk '{printf("%.4f ", $8)}')
  printf "% 10s" $prob
done

echo
