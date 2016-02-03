#!/bin/bash


. ./path.sh

model=
phone_to_vad_map=

#output format
. utils/parse_options.sh

if [ $# != 2 ]; then
  echo "Usage: $0 <lattice-dir> <vad-dir>"
fi

lats_dir=$1
vad_dir=$2

[ -z "$model" ] && model=$lats_dir/../final.mdl

mkdir -p $vad_dir

lattice-best-path --acoustic-scale=0.1 "ark:gunzip -c ${lats_dir}/lat.*.gz |" \
  ark,t:/dev/null ark:- | ali-to-phones --per-frame ${model} ark:- ark,t:- | \
  utils/int2sym.pl -f 2- data/lang/phones.txt 2>$vad_dir/phones.ali.log >$vad_dir/phones.ali || exit 1;

#####
# Post-processing, 
if [ -z "$phone_to_vad_map" ]; then
## map sil/<eps> to 0 and other phones to 1

phone_to_vad_map="$vad_dir/phone2vad.map"
(
while read line; do
  phone=`echo $line | awk '{print $1}'`
  if [ $phone == "sil" ] || [ $phone == "<eps>" ]; then
    echo $phone 0
  else
    echo $phone 1
  fi

done < data/lang/phones.txt 
)> $phone_to_vad_map

fi
#####

utils/sym2int.pl -f 2- $vad_dir/phone2vad.map $vad_dir/phones.ali 2>$vad_dir/vad.ali.log >$vad_dir/vad.ali || exit 1;

# TODO
# Smooth the vad
# smooth_vad_ali.py $vad_dir/vad.ali >$vad_dir/smoothed.vad.ali


steps/vad_scripts/ali_to_but_format.py $vad_dir/vad.ali >$vad_dir/vad.scp  


