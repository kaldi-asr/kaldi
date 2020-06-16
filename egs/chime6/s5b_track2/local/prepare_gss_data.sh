#!/bin/bash -u
# Copyright  2020    Prisyach Tatyana (STC-innovations Ltd)

. ./utils/parse_options.sh
. ./path.sh
. ./cmd.sh

echo >&2 "$0" "$@"
if [ $# -ne 3 ] ; then
  echo >&2 "$0" "$@"
  echo >&2 "$0: Error: wrong number of arguments"
  echo -e >&2 "Usage:\n  $0 <gss-dir> <output-data-dir>"
  echo -e >&2 "eg:\n  $0 enhanced_dir/audio/dev data/dev_gss"
  exit 1
fi

nj=8

gss_dir=$1
src_dir=$2
dir=$3

wav_list=$(find -L $gss_dir -name "*.wav" -printf "%f")
if [ ! -d $dir ]; then
  mkdir ${dir}
fi
echo $wav_list | awk -F ".wav" -v gss_dir=$gss_dir '{for (i=1; i<=NF; i++) {print $i" "gss_dir"/"$i".wav";}}' > $dir/wav_temp.scp
sort -u $dir/wav_temp.scp > $dir/wav.scp
rm -f $dir/wav_temp.scp
cp $src_dir/utt2spk $dir/utt2spk

utils/fix_data_dir.sh $dir

echo "$0 extracting mfcc freatures using segments file"
steps/make_mfcc.sh --mfcc-config conf/mfcc_hires.conf --nj $nj --cmd "$decode_cmd" ${dir}
steps/compute_cmvn_stats.sh ${dir}
cp $src_dir/text ${dir}/text
