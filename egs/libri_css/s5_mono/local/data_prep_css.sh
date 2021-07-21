#!/usr/bin/env bash
#
# Copyright  2020  Johns Hopkins University (Author: Desh Raj)
# Apache 2.0

# Begin configuration section.
# End configuration section
data_affix=
volume=1

. ./utils/parse_options.sh  # accept options

. ./path.sh

echo >&2 "$0" "$@"
if [ $# -ne 2 ] ; then
  echo >&2 "$0" "$@"
  echo >&2 "$0: Error: wrong number of arguments"
  echo -e >&2 "Usage:\n  $0 [opts] <corpus-dir> <wav-dile-dir>"
  echo -e >&2 "eg:\n  $0 /export/corpora/LibriCSS /export/c01/zhuc/css_data"
  exit 1
fi

corpus_dir=$1
wav_files_dir=$2

set -e -o pipefail

# If data is not already present, then download and unzip
if [ ! -d $corpus_dir/for_release ]; then
    echo "Downloading and unpacking LibriCSS data."    
    CWD=`pwd`
    mkdir -p $corpus_dir

    cd $corpus_dir

    # Download the data. If the data has already been downloaded, it
    # does nothing. (See wget -c) 
    wget -c --load-cookies /tmp/cookies.txt \
      "https://docs.google.com/uc?export=download&confirm=$(wget --quiet \
      --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate \
      'https://docs.google.com/uc?export=download&id=1Piioxd5G_85K9Bhcr8ebdhXx0CnaHy7l' \
      -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1Piioxd5G_85K9Bhcr8ebdhXx0CnaHy7l" \
      -O for_release.zip && rm -rf /tmp/cookies.txt

    # unzip (skip if already extracted)
    unzip -n for_release.zip

    # segmentation
    cd for_release
    python3 segment_libricss.py -data_path .

    cd $CWD
fi

# Process the downloaded data directory to get data in Kaldi format
# We first copy all the separated wav files from the original location
# without any directory structure. Here, the wav naming convention is 
# similar to that in the LibriCSS corpus meeting directories, with an
# additional `channel_n` at the end denoting the stream number, e.g.
# overlap_ratio_10.0_sil0.1_1.0_session0_actual10.1_channel_0.wav
# note that this "channel" is actually one of the separated audio streams.
mkdir -p data/local/data${data_affix}/wavs_orig
find $wav_files_dir -name '*.wav' -exec cp {} data/local/data${data_affix}/wavs_orig \;
local/prepare_data_css.py --srcpath $corpus_dir/for_release --wav-path data/local/data${data_affix}/wavs_orig \
  --tgtpath data/local/data${data_affix} --volume $volume

# Create dev and eval splits based on sessions. In total we have 10 sessions (session0 to 
# session9) of approximately 1 hour each.
dev_sessions="session0"
eval_sessions="session[1-9]"

mkdir -p data/dev${data_affix}
for file in wav.scp utt2spk text segments; do
  grep $dev_sessions data/local/data${data_affix}/$file | sort > data/dev${data_affix}/$file 
done

mkdir -p data/eval${data_affix}
for file in wav.scp utt2spk text segments; do
  grep $eval_sessions data/local/data${data_affix}/$file | sort > data/eval${data_affix}/$file 
done

# Move the utt2spk, segments, and text file to .bak so that they are only used
# in the last scoring stage. We also prepare a dummy utt2spk and spk2utt for
# these.
for datadir in dev eval; do
  for file in text utt2spk segments; do
    mv data/$datadir${data_affix}/$file data/$datadir${data_affix}/$file.bak
  done

  awk '{print $1, $1}' data/$datadir${data_affix}/wav.scp > data/$datadir${data_affix}/utt2spk
  utils/utt2spk_to_spk2utt.pl data/$datadir${data_affix}/utt2spk > data/$datadir${data_affix}/spk2utt

done
