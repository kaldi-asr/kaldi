#! /bin/bash

# Copyright 2016  Vimal Manohar
# Apache 2.0.

set -e
set -o pipefail

echo "$0 $@"  # Print the command line for logging

noise_word="<NOISE>"
spoken_noise_word="<SPOKEN_NOISE>"

. utils/parse_options.sh || exit 1;

. ./path.sh || exit 1;

if [ $# -ne 0 ]; then
  echo "Usage: $0"
  exit 1
fi

srcdir=data/local/data
tmpdir=data/local/

export PATH=$PATH:$KALDI_ROOT/tools/sph2pipe_v2.5

###############################################################################
# Format 1996 English Broadcast News Train (HUB4)
###############################################################################
mkdir -p data/train_bn96

local/data_prep/format_1996_bn_data.pl \
  $srcdir/train_bn96/audio.list $srcdir/train_bn96/transcripts.txt \
  data/train_bn96 || exit 1

mv data/train_bn96/text data/train_bn96/text.unnorm
local/data_prep/normalize_bn96_transcripts.pl $noise_word $spoken_noise_word \
  < data/train_bn96/text.unnorm > data/train_bn96/text

###############################################################################
# Format 1997 English Broadcast News Train (HUB4)
###############################################################################
mkdir -p data/train_bn97

local/data_prep/format_1997_bn_data.pl \
  $srcdir/train_bn97/audio.list $srcdir/train_bn97/transcripts.txt \
  data/train_bn97 || exit 1

mv data/train_bn97/text data/train_bn97/text.unnorm
local/data_prep/normalize_bn97_transcripts.pl $noise_word $spoken_noise_word \
  < data/train_bn97/text.unnorm > data/train_bn97/text

###############################################################################
# Format 1996 English Broadcast News Dev (HUB4)
###############################################################################
mkdir -p data/dev96pe 
mkdir -p data/dev96ue

cp $srcdir/hub4_96_dev_eval/dev96_uem_segments data/dev96ue/segments
cp $srcdir/hub4_96_dev_eval/dev96_uem_utt2spk data/dev96ue/utt2spk
cp $srcdir/hub4_96_dev_eval/dev96_uem_wav_scp data/dev96ue/wav.scp
cp $srcdir/hub4_96_dev_eval/dev96_uem_stm data/dev96ue/stm
cp $srcdir/hub4_96_dev_eval/glm data/dev96ue/glm

awk '{if ($4 > $3) print $0}' $srcdir/hub4_96_dev_eval/dev96_pem_segments \
  > data/dev96pe/segments
cp $srcdir/hub4_96_dev_eval/dev96_pem_utt2spk data/dev96pe/utt2spk
cp $srcdir/hub4_96_dev_eval/dev96_pem_wav_scp data/dev96pe/wav.scp
cp $srcdir/hub4_96_dev_eval/dev96_pem_stm data/dev96pe/stm
cp $srcdir/hub4_96_dev_eval/glm data/dev96pe/glm

###############################################################################
# Format 1996 English Broadcast News Eval (HUB4)
###############################################################################
mkdir -p data/eval96
mkdir -p data/eval96.pem 

cp $srcdir/hub4_96_dev_eval/eval96_pem_segments data/eval96.pem/segments
cp $srcdir/hub4_96_dev_eval/eval96_pem_utt2spk data/eval96.pem/utt2spk
cp $srcdir/hub4_96_dev_eval/eval96_wav_scp data/eval96.pem/wav.scp
cp $srcdir/hub4_96_dev_eval/eval96_stm data/eval96.pem/stm
cp $srcdir/hub4_96_dev_eval/glm data/eval96.pem/glm

cp $srcdir/hub4_96_dev_eval/eval96_uem_segments data/eval96/segments
cp $srcdir/hub4_96_dev_eval/eval96_uem_utt2spk data/eval96/utt2spk
cp $srcdir/hub4_96_dev_eval/eval96_wav_scp data/eval96/wav.scp
cp $srcdir/hub4_96_dev_eval/eval96_stm data/eval96/stm
cp $srcdir/hub4_96_dev_eval/glm data/eval96/glm

###############################################################################
# Format 1997-98 Hub4 Broadcast news evalutation
###############################################################################
for t in eval97 eval98; do
  mkdir -p data/$t data/${t}.pem
  cp $srcdir/$t/segments data/$t/segments
  cp $srcdir/$t/utt2spk data/$t/utt2spk
  cp $srcdir/$t/segments.pem data/${t}.pem/segments
  cp $srcdir/$t/utt2spk.pem data/${t}.pem/utt2spk
  cp $srcdir/$t/wav.scp data/$t/wav.scp
  cp $srcdir/$t/wav.scp data/${t}.pem/wav.scp
  cp $srcdir/$t/stm data/$t/stm
  cp $srcdir/$t/stm data/${t}.pem/stm
  cp $srcdir/$t/glm data/$t/glm
  cp $srcdir/$t/glm data/${t}.pem/glm
done

###############################################################################
# Format 1999 Hub4 Broadcast news evalutation
###############################################################################
for d in eval99_1 eval99_2; do
  mkdir -p data/${d} data/${d}.pem
  cp $srcdir/eval99/${d}_uem_segments data/${d}/segments
  cp $srcdir/eval99/${d}_uem_utt2spk data/${d}/utt2spk
  cp $srcdir/eval99/${d}_pem_segments data/${d}.pem/segments
  cp $srcdir/eval99/${d}_pem_utt2spk data/${d}.pem/utt2spk
  cp $srcdir/eval99/${d}_wav_scp data/${d}/wav.scp
  cp $srcdir/eval99/${d}_wav_scp data/${d}.pem/wav.scp
  cp $srcdir/eval99/${d}_stm data/${d}/stm
  cp $srcdir/eval99/${d}_stm data/${d}.pem/stm
  cp $srcdir/eval99/${d}_glm data/${d}/glm
  cp $srcdir/eval99/${d}_glm data/${d}.pem/glm
done

for d in train_bn96 train_bn97 eval96 eval96.pem dev96pe dev96ue eval97 eval97.pem \
         eval98 eval98.pem eval99_1 eval99_1.pem eval99_2 eval99_2.pem; do
  utils/utt2spk_to_spk2utt.pl data/$d/utt2spk > data/$d/spk2utt
  awk '{print $1" "$1" 1"}' data/${d}/wav.scp > \
    data/${d}/reco2file_and_channel
  utils/fix_data_dir.sh data/${d}
done

utils/combine_data.sh data/train data/train_bn96 data/train_bn97
