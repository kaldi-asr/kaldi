#!/bin/bash

# this script creates a new data directory data/sdm1/train_cleanali or
# data/mdm8/train_cleanali which has the segment ids from (e.g.) data/sdm1/train
# but the wav data is copied from data/ihm.  This is a little tricky because the
# utterance ids are different between the different mics


train_set=train

. utils/parse_options.sh

if [ $# != 1 ]; then
  echo "Usage: $0 [sdm1|mdm8]"
  exit 1
fi

mic=$1

if [ $mic == "ihm" ]; then
  echo "$0: this script does not make sense for ihm mic."
  exit 1;
fi

. ./cmd.sh
. ./path.sh

for f in data/ihm/${train_set}/utt2spk data/$mic/${train_set}/utt2spk; do
  if [ ! -f $f ]; then
    echo "$0: expected file $f to exist"
    exit 1
  fi
done

set -e -o pipefail

mkdir -p data/$mic/${train_set}_ihmdata

# the utterance-ids and speaker ids will be from the SDM or MDM data
cp data/$mic/${train_set}/{spk2utt,text,utt2spk} data/$mic/${train_set}_ihmdata/
# the recording-ids will be from the IHM data.
cp data/ihm/${train_set}/{wav.scp,reco2file_and_channel} data/$mic/${train_set}_ihmdata/

# map sdm/mdm segments to the ihm segments

mic_base_upcase=$(echo $mic | sed 's/[0-9]//g' | tr 'a-z' 'A-Z')

# the ihmutt2utt file maps from the IHM utterance-id to the SDM or MDM utterance-id.
# It has lines like:
# AMI_EN2001a_H02_FEO065_0021133_0021442 AMI_EN2001a_SDM_FEO065_0021133_0021442

tmpdir=data/$mic/${train_set}_ihmdata/

awk '{print $1, $1}' <data/ihm/${train_set}/utt2spk | \
  sed -e "s/_H[0-9][0-9]_/_${mic_base_upcase}_/" | \
  awk '{print $2, $1}' >$tmpdir/ihmutt2utt

# Map the 1st field of the segments file from the ihm data (the 1st field being
# the utterance-id) to the corresponding SDM or MDM utterance-id.  The other
# fields remain the same (e.g. we want the recording-ids from the IHM data).
utils/apply_map.pl -f 1 $tmpdir/ihmutt2utt <data/ihm/${train_set}/segments >data/$mic/${train_set}_ihmdata/segments

utils/fix_data_dir.sh data/$mic/${train_set}_ihmdata

rm $tmpdir/ihmutt2utt

exit 0;
