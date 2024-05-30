#!/usr/bin/env bash
# Copyright 2015  Johns Hopkins University (Author: Vijayaditya Peddinti)
# Apache 2.0.
set -e
stage=0
# Location of aspire data.
aspire_data=/export/corpora/LDC/LDC2017S21/IARPA-ASpIRE-Dev-Sets-v2.0/data  # for JHU

mean_rms=0.0417 # determined from the mean rms value of data/train_rvb/mean_rms
. ./path.sh # Needed for KALDI_ROOT

. utils/parse_options.sh

dev_transcript=$aspire_data/dev_and_dev_test_STM_files
dev_audio=$aspire_data/dev_and_dev_test_audio/ASpIRE_single_dev
test_audio=$aspire_data/dev_and_dev_test_audio/ASpIRE_single_dev_test
if [ ! -f $aspire_data/my_english.glm ]; then
  echo "Expected to find the glm file, provided in ASpIRE challenge."
  echo "Please provide the glm file in $aspire_data." && exit 1;
fi

# (1) Get transcripts in one file, and clean them up ..
tmpdir=`pwd`/data/local/data
mkdir -p $tmpdir
if [ $stage -le 0 ]; then

  find $dev_transcript/ -name 'dev.stm'  > $tmpdir/transcripts.flist
  find $dev_audio/ -name '*.wav'  > $tmpdir/wav.flist
  find $test_audio/ -name '*.wav'  > $tmpdir/wav_test.flist

  n=$(awk '{print $1}' $(cat $tmpdir/transcripts.flist) | uniq | wc -l)
  if [ $n -ne 30 ]; then
    echo "Expected to find 30 transcript files in the aspire_single_dev_transcript directory, found $n"
    exit 1;
  fi
  n=`cat $tmpdir/wav.flist | wc -l`
  if [ $n -ne 30 ]; then
    echo "Expected to find 30 .wav files in the aspire_single_dev directory, found $n"
    exit 1;
  fi
  n=`cat $tmpdir/wav_test.flist | wc -l`
  if [ $n -ne 60 ]; then
    echo "Expected to find 60 .wav files in the aspire_single_dev_test data, found $n"
    exit 1;
  fi
fi

# create the dev_aspire files
dev=data/dev_aspire
if [ $stage -le 1 ]; then
  mkdir -p $dev

# transcription file format
# single_074f59de 1 single_074f59de 497.775 506.595 um everybody can't get their needs met in in in in a in a negotiations or to to their satisfaction but at least you're attemptin
  
  echo -n > $tmpdir/text.1 || exit 1;
  
  python -c "
import sys
trans_file = open('$tmpdir/text.1', 'w')
utt2spk_file = open('$dev/utt2spk', 'w')
segments_file = open('$dev/segments', 'w')
stm_file = open('$dev/stm', 'w')
utt2spk = []

for file_name in open('$tmpdir/transcripts.flist', 'r').readlines():
  lines = open(file_name.strip()).readlines()
  for line in lines:
    parts = line.split()
    file_id = parts[0]
    utt_id = '{0}-{1}-{2:06}-{3:06}'.format(parts[0], parts[1], int(float(parts[3]) * 1000), int(float(parts[4]) * 1000))
    spk_id = '{0}-{1}'.format(parts[0], parts[1])
    stm_file.write('{0} A {0} {1}\n'.format(spk_id, ' '.join(parts[3:]))) 
    trans_file.write('{0} {1}\n'.format(utt_id, ' '.join(parts[5:])))
    utt2spk.append(('{0} {1}\n'.format(utt_id, spk_id)))
    segments_file.write('{0} {1}-1 {2} {3}\n'.format(utt_id, file_id, parts[3], parts[4]))
stm_file.close()
trans_file.close()
utt2spk.sort()
utt2spk_file.write(''.join(utt2spk))
utt2spk_file.close()
segments_file.close()
" || exit 1; 
fi

if [ $stage -le 2 ]; then
  sort $tmpdir/text.1 | grep -v '((' | \
    awk '{if (NF > 1){ print; }}' | \
    sed 's:\[laugh\]:[laughter]:g' | \
    sed 's:\[sigh\]:[noise]:g' | \
    sed 's:\[cough\]:[noise]:g' | \
    sed 's:\[sigh\]:[noise]:g' | \
    sed 's:\[mn\]:[noise]:g' | \
    sed 's:\[breath\]:[noise]:g' | \
    sed 's:\[lipsmack\]:[noise]:g' > $tmpdir/text.2
  cp $tmpdir/text.2 $dev/text

  utils/utt2spk_to_spk2utt.pl <$dev/utt2spk > $dev/spk2utt
fi

if [ $stage -le 3 ]; then
  for f in `cat $tmpdir/wav.flist`; do
    # convert to absolute path
    utils/make_absolute.sh $f
  done > $tmpdir/wav_abs.flist
  
  cat $tmpdir/wav_abs.flist | python -c "
import sys, os, subprocess, re

for line in sys.stdin.readlines():
  if len(line.strip()) == 0:
    continue
  proc = subprocess.Popen('sox {0} -n stat'.format(line.strip()).split(), stdout = subprocess.PIPE, stderr = subprocess.PIPE)
  out, err = proc.communicate()
  out_rms = $mean_rms/float(re.split('RMS\s+amplitude:', err)[1].split()[0])
  line = line.strip()
  file_id=os.path.splitext(os.path.split(line)[1])[0]+'-1'
  print '{0} sox --vol {1} {2} -r 8000 -t wav - |'.format(file_id, out_rms, line)
"| sort -k1,1 -u  > $dev/wav.scp || exit 1;
  cat $dev/wav.scp |awk '{printf("%s %s A\n", $1, $1)}' > $dev/reco2file_and_channel
  cp $aspire_data/my_english.glm $dev/glm
fi

# prepare test data
if [ $stage -le 4 ]; then
  for dataset in test ; do
    test=data/${dataset}_aspire
    mkdir -p $test
    for f in `cat $tmpdir/wav_${dataset}.flist`; do
      # convert to absolute path
      utils/make_absolute.sh $f
    done > $tmpdir/wav_${dataset}_abs.flist
    cat $tmpdir/wav_${dataset}_abs.flist | \
    python -c "
import sys, os, subprocess, re

lines = sys.stdin.readlines()
for line in lines:
  if len(line.strip()) == 0:
    continue
  proc = subprocess.Popen('sox {0} -n stat'.format(line.strip()).split(), stdout = subprocess.PIPE, stderr = subprocess.PIPE)
  out, err = proc.communicate()
  out_rms = $mean_rms/float(re.split('RMS\s+amplitude:', err)[1].split()[0])
  line = line.strip()
  file_id=os.path.splitext(os.path.split(line)[1])[0]+'-1'
  print '{0} sox --vol {1} {2} -r 8000 -t wav - |'.format(file_id, out_rms, line)
    " | sort -k1,1 -u  > $test/wav.scp || exit 1;

    cat $test/wav.scp |awk '{printf("%s %s\n", $1, $1)}' > $test/utt2spk
    cat $test/wav.scp |awk '{printf("%s %s\n", $1, $1)}' > $test/spk2utt
    cat $test/wav.scp |awk '{printf("%s %s A\n", $1, $1)}' > $test/reco2file_and_channel
    cp $aspire_data/my_english.glm $test/glm
  done
fi

echo "Aspire dev/test/eval data preparation succeeded"
