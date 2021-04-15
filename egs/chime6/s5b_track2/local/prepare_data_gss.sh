#!/bin/bash
#
# Copyright  2017  Johns Hopkins University (Author: Shinji Watanabe, Yenda Trmal)
# Apache 2.0

# Begin configuration section.
mictype=worn # worn, ref or others
cleanup=true
# End configuration section
. ./utils/parse_options.sh  # accept options.. you can run this run.sh with the

. ./path.sh

echo >&2 "$0" "$@"
if [ $# -ne 5 ] ; then
  echo >&2 "$0" "$@"
  echo >&2 "$0: Error: wrong number of arguments"
  echo -e >&2 "Usage:\n  $0 [opts] <audio-dir> <output-dir>"
  echo -e >&2 "eg:\n  $0 /corpora/chime5/audio/train data/train"
  exit 1
fi

set -e -o pipefail

adir=$(utils/make_absolute.sh $1)
dir=$2
rttm_dir=$3
rttm_file=$4
data_in_gss=$5

mkdir -p $dir

find -L $adir -name  "S[0-9]*.wav" | \
  perl -ne '{
    chomp;
    $path = $_;
    next unless $path;
    @F = split "/", $path;
    ($f = $F[@F-1]) =~ s/.wav//;
    print "$f $path\n";
  }' | sort > $dir/wav.list

$cleanup && rm -f $dir/text.* $dir/wav.scp.* $dir/wav.flist

cp $rttm_dir/${rttm_file} $rttm_dir/${rttm_file}_1
sed -i 's/'.ENH'//g' $rttm_dir/${rttm_file}_1
local/convert_rttm_to_utt2spk_and_segments.py --append-reco-id-to-spkr=true $rttm_dir/${rttm_file}_1 \
  <(awk '{print $2".ENH "$2" "$3}' ${rttm_dir}/${rttm_file}_1 |sort -u) \
  ${dir}/utt2spk ${dir}/segments_full

local/truncate_rttm.py $rttm_dir/${rttm_file}_1 local/uem_file $rttm_dir/${rttm_file}_introduction_removed
local/convert_rttm_to_utt2spk_and_segments.py --append-reco-id-to-spkr=true $rttm_dir/${rttm_file}_introduction_removed \
  <(awk '{print $2".ENH "$2" "$3}' $rttm_dir/${rttm_file}_introduction_removed |sort -u) \
  ${dir}/utt2spk ${dir}/segments_select_1

paste ${dir}/segments_select_1 $dir/wav.list | awk '{print $1 " " $6}' > $dir/wav_full.scp
awk 'NR==FNR{seen[$1]; next} $1 in seen' ${data_in_gss}/wav.scp ${dir}/wav_full.scp > ${dir}/wav.scp

#awk 'NR==FNR{seen[$1]; next} $1 in seen' ${dir}/segments_select ${dir}/wav_full.scp > $dir/wav.scp

awk '{split($1, lst, "-"); spk=lst[1]"-"lst[2]; print($1, spk)}' $dir/wav.scp > $dir/utt2spk
#cut -f 1 -d ' ' $dir/wav.scp | \
#  perl -ne 'chomp;$utt=$_;s/-.*//;print "$utt $_\n";' > $dir/utt2spk

utils/utt2spk_to_spk2utt.pl $dir/utt2spk > $dir/spk2utt

# Check that data dirs are okay!
utils/validate_data_dir.sh --no-text --no-feats $dir || exit 1
