#!/usr/bin/env bash

# Copyright 2013  Daniel Povey
# Apache 2.0.

if [ $# -ne 2 ]; then
  echo "Usage: $0 <path-to-LDC2011S08> <path-to-output>";
  echo "e.g. $0 /export/corpora5/LDC/LDC2011S08 data";
  exit 1;
fi

src=$1
d=$2
  

for condition in short3; do # could also add "10sec" and "summed" here.
  sph_src=$src/data/test/data/$condition
  if [ ! -d $sph_src ]; then
    echo "$0: expecting directory $sph_src to exist"
    exit 1;
  fi

  data=$d/sre08_test_${condition}
  mkdir -p $data
  for f in $sph_src/*.sph; do
    base=$(basename $f | sed s:.sph$::)
    for side in A B; do
      if [ $side == "A" ]; then 
        channel=1
      else
        channel=2
      fi
      utt_id=${base}_${side} # e.g. thagc_B
      echo "${utt_id} sph2pipe -f wav -p -c ${channel} $f |"
    done
  done | sort > $data/wav.scp
  ! [ -s $data/wav.scp ] && echo "$0: Error creating wav.scp (empty output)" && exit 1;

  # We don't have speaker information here, so we just make the utt2spk a one-to-one
  # mapping (this file is required by certain Kaldi scripts)
  cat $data/wav.scp | awk '{print $1, $1}' | tee $data/spk2utt > $data/utt2spk
  
  # Use the "trials" file to get the gender
  cat $src/data/trials/*-${condition}.ndx  | sed 's/.sph:/_/' | awk '{print $3, $2}' | sort | uniq > $data/spk2gender

  # Note: not all of these utterances appear in the trials file, e.g. the interview
  # segments have a "B" side which is never used.  So the spk2gender file is smaller
  # than the spk2utt file.  When we do fix_data_dir.sh, the un-needed ones get removed.
  utils/fix_data_dir.sh $data || exit 1;
  utils/validate_data_dir.sh --no-text --no-feats $data || exit 1;

  # Filter into male and female parts, since we have gender-dependent
  # processing.
  echo "Creating female subset of $data"
  cat $data/spk2gender | grep -w f > spklist
  utils/subset_data_dir.sh --spk-list spklist $data ${data}_female
  echo "Creating male subset of $data"
  cat $data/spk2gender | grep -w m > spklist
  utils/subset_data_dir.sh --spk-list spklist $data ${data}_male
done


trials=$d/sre08_trials
mkdir -p $trials

tail -n +2 $src/data/keys/NIST_SRE08_KEYS.v0.1/trial-keys/NIST_SRE08_short2-short3.trial.key | \
  sed 's:,b,:_B,:; s:,a,:_A,:; s:,: :g' > $trials/short2-short3.trials

cat $trials/short2-short3.trials | awk '{print $2, $0}' | \
  utils/filter_scp.pl $d/sre08_test_short3_female/utt2spk | cut -d ' ' -f 2- \
  > $trials/short2-short3-female.trials
cat $trials/short2-short3.trials | awk '{print $2, $0}' | \
  utils/filter_scp.pl $d/sre08_test_short3_male/utt2spk | cut -d ' ' -f 2- \
  > $trials/short2-short3-male.trials

n1=$(cat $trials/short2-short3.trials | wc -l)
n2=$(cat $trials/short2-short3-{male,female}.trials | wc -l)
if ! [ $n1 -eq $n2 ]; then
  echo "Error: length mismatch (missing data?) $n1 != $n2"
  exit 1
fi
exit 0

