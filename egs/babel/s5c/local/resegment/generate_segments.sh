#!/usr/bin/env bash

# Copyright 2014  Vimal Manohar, Johns Hopkins University (Author: Jan Trmal)
# Apache 2.0

set -o pipefail
set -e

nj=8
cmd=run.pl
stage=0
segmentation_opts="--isolated-resegmentation --min-inter-utt-silence-length 1.0 --silence-proportion 0.05"
decoder_extra_opts=""
reference_rttm=
get_text=false  # Get text corresponding to new segments in ${output_dir}
                # Assuming text is in $data/$type directory.
                # Does not work very well because the data does not get aligned to many training transcriptions.
noise_oov=false     # Treat <oov> as noise instead of speech
beam=7.0
max_active=1000

#debugging stuff
echo $0 $@

[ -f ./path.sh ] && . ./path.sh
. parse_options.sh || exit 1;

set -u

if [ $# -ne 5 ]; then
  echo "Usage: $0 [options] <data-dir> <lang-dir> <model-dir> <temp-dir>"
  echo " Options:"
  echo "    --cmd (run.pl|queue.pl...)      # specify how to run the sub-processes."
  echo "    --nj <numjobs>          # Number of parallel jobs. "
  echo "                              For the standard data directories of dev10h, dev2h and eval"
  echo "                              this is taken from the lang.conf file"
  echo "    --segmentation-opts '--opt1 opt1val --opt2 opt2val' # options for segmentation.py"
  echo "    --reference-rttm        # Reference RTTM file that will be used for analysis of the segmentation"
  echo "    --get-text (true|false) # Convert text from base data directory to correspond to the new segments"
  echo
  echo "e.g.:"
  echo "$0 data/dev10h data/lang exp/tri4b_seg exp/tri4b_resegment_dev10h"
  exit 1
fi

datadir=$1      # The base data directory that contains at least the files wav.scp and reco2file_and_channel
lang=$2
model_dir=$3    # Segmentation model directory created using local/resegment/run_segmentation_train.sh
temp_dir=$4     # Temporary directory to store some intermediate files during segmentation
output_dir=$5   # The target directory

###############################################################################
#
# Phone Decoder
#
###############################################################################

mkdir -p $temp_dir
dirid=`basename $datadir`
total_time=0
t1=$(date +%s)

if [ $stage -le 0 ] ; then
  steps/decode_nolats.sh ${decode_extra_opts+} --write-words false --write-alignments true \
    --cmd "$cmd" --nj $nj --beam $beam --max-active $max_active \
    $model_dir/phone_graph $datadir $model_dir/decode_${dirid} || exit 1
fi

if [ $stage -le 1 ]; then
  [ ! -f  $model_dir/decode_${dirid}/ali.1.gz ] && echo "File $model_dir/decode_${dirid}/ali.1.gz does not exist!" && exit 1
  $cmd JOB=1:$nj $model_dir/decode_${dirid}/log/predict.JOB.log \
    gunzip -c $model_dir/decode_${dirid}/ali.JOB.gz \| \
    ali-to-phones --per-frame=true $model_dir/final.mdl ark:- ark,t:- \| \
    utils/int2sym.pl -f 2- $lang/phones.txt \| \
    gzip -c '>' $temp_dir/pred.JOB.gz || exit 1

  mkdir -p $temp_dir/pred
  gunzip -c $temp_dir/pred.*.gz | \
    perl -ne '($file, $phones)=split / /, $_, 2;
              open($fh, ">'$temp_dir/pred/'$file.pred" ) or die $!;
              print {$fh} "$file $phones";
              close($fh);' || exit 1

fi
t2=$(date +%s)
total_time=$((total_time + t2 - t1))
echo "SI decoding done in $((t2-t1)) seconds"


###############################################################################
#
# Resegmenter
#
###############################################################################

if ! [ `cat $lang/phones/optional_silence.txt | wc -w` -eq 1 ]; then
  echo "Error: this script only works if $lang/phones/optional_silence.txt contains exactly one entry.";
  echo "You'd have to modify the script to handle other cases."
  exit 1;
fi

silphone=`cat $lang/phones/optional_silence.txt`
# silphone will typically be "sil" or "SIL".

# 3 sets of phones: 0 is silence, 1 is noise, 2 is speech.,
(
echo "$silphone 0"
if ! $noise_oov; then
  grep -v -w $silphone $lang/phones/silence.txt \
    | awk '{print $1, 1;}' \
    | sed 's/SIL\(.*\)1/SIL\10/' \
    | sed 's/<oov>\(.*\)1/<oov>\12/'
else
  grep -v -w $silphone $lang/phones/silence.txt \
    | awk '{print $1, 1;}' \
    | sed 's/SIL\(.*\)1/SIL\10/'
fi
cat $lang/phones/nonsilence.txt | awk '{print $1, 2;}' | sed 's/\(<.*>.*\)2/\11/' | sed 's/<oov>\(.*\)1/<oov>\12/'
) > $temp_dir/phone_map.txt

mkdir -p $output_dir
mkdir -p $temp_dir/log

local/resegment/segmentation.py --verbose 2 $segmentation_opts \
  $temp_dir/pred $temp_dir/phone_map.txt 2>$temp_dir/log/resegment.log | \
  sort > $output_dir/segments || exit 1

if [ ! -s $output_dir/segments ] ; then
  echo "Zero segments created during segmentation process."
  echo "That means something failed. Try the cause and re-run!"
  exit 1
fi

t2=$(date +%s)
total_time=$((total_time + t2 - t1))
echo "Resegment data done in $((t2-t1)) seconds"

for file in reco2file_and_channel wav.scp ; do
  [ ! -f $datadir/$file ] && echo "Expected file $datadir/$file to exist" && exit 1
  cp $datadir/$file $output_dir/$file
done

# We'll make the speaker-ids be the same as the recording-ids (e.g. conversation
# sides).  This will normally be OK for telephone data.
cat $output_dir/segments | awk '{print $1, $2}' > $output_dir/utt2spk || exit 1
utils/utt2spk_to_spk2utt.pl ${output_dir}/utt2spk > $output_dir/spk2utt || exit 1


dur_hours=`cat ${output_dir}/segments | awk '{num_secs += $4 - $3;} END{print (num_secs/3600);}'`
echo "Extracted segments of total length of $dur_hours hours audio"

echo ---------------------------------------------------------------------
echo "Resegment data Finished successfully on" `date`
echo ---------------------------------------------------------------------

exit 0
