#!/usr/bin/env bash
#
# Copyright  2020  Johns Hopkins University (Author: Desh Raj)
# Apache 2.0

# Begin configuration section.
# End configuration section

data_affix= # if provided, simulated data with this affix will be used
forced_alignments= # path to forced alignments file. If provided, the original
                   # segments and utt2spk files will be replaced with these new
                   # files obtained using forced alignments. Also, the existing
                   # reference text file will be removed since we assume this
                   # is for a diarization like system.

. ./utils/parse_options.sh  # accept options

. ./path.sh

echo >&2 "$0" "$@"
if [ $# -ne 2 ] ; then
  echo >&2 "$0" "$@"
  echo >&2 "$0: Error: wrong number of arguments"
  echo -e >&2 "Usage:\n  $0 [opts] <corpus-dir> <librispeech-dir>"
  echo -e >&2 "eg:\n  $0 /export/corpora/LibriCSS /export/corpora/LibriSpeech"
  exit 1
fi

corpus_dir=$1
librispeech_dir=$2

set -e -o pipefail

if ! [ -d $corpus_dir ]; then
  echo "$0: $corpus_dir does not exist. Please run the data simulation first."
  exit 1
fi

for dataset in train; do
  echo "$0: Preparing $dataset data.."
  output_data_dir=data/${dataset}_sim${data_affix}
  if [ -d $output_data_dir ]; then
    echo "$0: $output_data_dir already exists. Please remove to continue."
    exit 1
  fi
  wav_data_dir=$corpus_dir/data/SimLibriCSS-${dataset}${data_affix}/
  mkdir -p ${output_data_dir}
  local/prepare_simulated_meetings_data.py --txtpath $librispeech_dir \
    --wavpath $wav_data_dir --tgtpath $output_data_dir --type $dataset
  utils/fix_data_dir.sh ${output_data_dir}

  # also fix the wav_clean.scp
  file=$output_data_dir/wav_clean.scp
  sort -k1,1 -u <$file >$file.tmp
  if ! cmp -s $file $file.tmp; then
    echo "$0: file $1 is not in sorted order or not unique, sorting it"
    mv $file.tmp $file
  else
    rm $file.tmp
  fi
done

if ! [ -z "$forced_alignments" ]; then
  alignments_dir=`basename $forced_alignments .tar.gz`
  if ! [ -d $alignments_dir ]; then
    tar -xvzf $forced_alignments
  fi
  for dataset in train dev test; do
    echo "$0: generating $dataset RTTM from forced alignments"
    output_data_dir=data/${dataset}_sim

    # Store original segments and utt2spk in backup
    mv $output_data_dir/utt2spk $output_data_dir/utt2spk.bak
    mv $output_data_dir/segments $output_data_dir/segments.bak

    # Generate force aligned RTTM using the alignments
    cat $alignments_dir/${dataset}* > tmp.ctm
    local/generate_forced_aligned_rttm.py tmp.ctm $output_data_dir/utt2spk.bak \
      $output_data_dir/segments.bak > $output_data_dir/rttm.forced
    rm tmp.ctm
    
    # Get new segments and utt2spk from created RTTM
    rttm_file=$output_data_dir/rttm.forced
    local/convert_rttm_to_utt2spk_and_segments.py \
      --append-reco-id-to-spkr true $rttm_file \
      <(awk '{print $2" "$2" "$3}' $rttm_file |sort -u) \
      $output_data_dir/utt2spk $output_data_dir/segments
    rm $output_data_dir/text 2> /dev/null
  done
fi

exit 0