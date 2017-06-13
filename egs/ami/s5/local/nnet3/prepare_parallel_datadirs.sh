#!/bin/bash

# this script creates a new data directory data/$new_mic
# where the train, dev and eval directories are copied from $original_mic
# in addition to these a new data directory train_parallel is created which has
# the segment ids from data/$original_mic but the wav data is copied from 
# data/$parallel_mic

original_mic=sdm1
parallel_mic=ihm
new_mic=sdm1_cleanali

. cmd.sh
. ./path.sh
. ./utils/parse_options.sh

#copy the clean data directory and change the segment ids
for dset in train dev eval; do
  utils/copy_data_dir.sh data/$original_mic/$dset data/$new_mic/$dset
done
dset=train
utils/copy_data_dir.sh data/$parallel_mic/$dset data/$new_mic/${dset}_parallel
rm -rf data/$new_mic/${dset}_parallel/{text,feats.scp,cmvn.scp}
cp data/$new_mic/$dset/{spk2utt,text,utt2spk} data/$new_mic/${dset}_parallel
cp data/$new_mic/${dset}_parallel/wav.scp data/$new_mic/${dset}_parallel/wav.scp_full
cp data/$new_mic/${dset}_parallel/reco2file_and_channel data/$new_mic/${dset}_parallel/reco2file_and_channel_full

dset=train
# map sdm/mdm segments to the ihm segments
tmpdir=`mktemp -d ./tmpXXX`
cat data/$parallel_mic/$dset/segments | sed -e "s/_H[0-9][0-9]_//g" > $tmpdir/key2ihm
cat data/$new_mic/$dset/segments | awk '{print $1}' > $tmpdir/dm_utts
mic_basename=$(echo $original_mic | sed -e "s/[0-9]//g")
if [ $mic_basename == "sdm" ]; then
  pattern="_SDM_"
else
  pattern="_MDM_"
fi
cat $tmpdir/dm_utts | sed -e "s/$pattern//g" > $tmpdir/key
paste -d' ' $tmpdir/key $tmpdir/dm_utts  > $tmpdir/key2dm

python -c "
ihm = dict(map(lambda x: [x.split()[0], ' '.join(x.split()[1:])], open('$tmpdir/key2ihm').readlines()))
dm = dict(map(lambda x: x.split(), open('$tmpdir/key2dm').readlines()))

keys = ihm.keys()
keys.sort()

for key in keys :
  try:
    print '{0} {1}'.format(dm[key], ihm[key])
  except KeyError:
    continue
" > data/$new_mic/${dset}_parallel/segments
  
cat data/$new_mic/${dset}_parallel/segments | awk '{print $2}' |sort -u > $tmpdir/ids
utils/filter_scp.pl $tmpdir/ids \
  data/$new_mic/${dset}_parallel/wav.scp_full  > \
  data/$new_mic/${dset}_parallel/wav.scp

utils/filter_scp.pl $tmpdir/ids \
  data/$new_mic/${dset}_parallel/reco2file_and_channel_full  > \
  data/$new_mic/${dset}_parallel/reco2file_and_channel
utils/fix_data_dir.sh data/$new_mic/${dset}_parallel

exit 0;
