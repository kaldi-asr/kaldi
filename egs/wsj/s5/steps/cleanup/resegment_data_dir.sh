#!/bin/bash

# Modified based on the script: utils/data/copy_data_dir.sh 

# Copyright 2013  Johns Hopkins University (author: Daniel Povey)
# Apache 2.0

# begin configuration section
frame_shift=10   # In milliseconds
validate_opts=   # should rarely be needed.
# end configuration section

. utils/parse_options.sh

if [ $# != 4 ]; then
  echo "This script segments a data directory by segmenting the utterances "
  echo "using a <sub-segments> file and adds the corresponding text to it from <sub-segment-text>"
  echo "The sub-segments file has the format <new-utt-id> <old-utt-id> <s2> <e2>, where "
  echo "<new-utt-id> is an id for a segment of the old utterance <old-utt-id>"
  echo "<s2> is the start time of <new-utt-id> relative to start time of <old-utt-id>, s"
  echo "<e2> is the end time of <new-utt-id> relative to start time of <old-utt-id>, s"
  echo "The sub-segment-text file has the format <new-utt-id> <text>"
  echo "A new segments file will be created in the output data directory containing lines "
  echo "<new-utt-id> <reco-id> (<s2>+s) (<e2>+s)"
  echo
  echo "Usage: "
  echo "  $0 <srcdir> <sub-segment-text> <sub-segments> <destdir>"
  echo "e.g.:"
  echo " $0 data/train data/train/sub_segment_text data/train/sub_segments data/train_1"
  exit 1;
fi

export LC_ALL=C

srcdir=$1
sub_segment_text=$2
sub_segments=$3
destdir=$4

if [ ! -f $srcdir/utt2spk ]; then
  echo "resegment_data_dir.sh: no such file $srcdir/utt2spk"
  exit 1;
fi

set -e;

rm -r $destdir 2>/dev/null || true
mkdir -p $destdir

cat $srcdir/segments | awk '{print $1, $2}' > $destdir/utt2reco_map
cat $sub_segments | awk '{print $1, $2}' > $destdir/utt_new2utt_map

if [ -f $srcdir/utt2spk ]; then
  utils/apply_map.pl -f 2 $srcdir/utt2spk < $destdir/utt_new2utt_map >$destdir/utt2spk
else
  echo "$0: no such file $srcdir/utt2spk"
  exit 1
fi

if [ -f $srcdir/utt2uniq ]; then
  utils/apply_map.pl -f 2 $srcdir/utt2uniq < $destdir/utt_new2utt_map >$destdir/utt2uniq
fi

if [ ! -f $sub_segments ]; then
  echo "resegment_data_dir.sh: no such file $sub_segments"
  exit 1;
fi

if [ -f $srcdir/segments ]; then
  cp $srcdir/wav.scp $destdir
  
  python -c 'import sys
start_times = dict()
end_times = dict()
for line in open(sys.argv[1]):
  splits = line.strip().split()
  start_times[splits[0]] = float(splits[2])
  end_times[splits[0]] = float(splits[3])

for line in sys.stdin.readlines():
  splits = line.strip().split()
  beg = float(splits[2]) + start_times[splits[1]]
  end = min(float(splits[3]) + start_times[splits[1]], end_times[splits[1]])
  print ("%s %s %.02f %.02f" % (splits[0], splits[1], beg, end))' $srcdir/segments < $sub_segments | \
    utils/apply_map.pl -f 2 $destdir/utt2reco_map > $destdir/segments || exit 1
  
  if [ -f $srcdir/feats.scp ]; then
    feat-to-len scp:$srcdir/feats.scp ark,t:$destdir/len

    python -c 'import sys
feats = dict()
for line in open(sys.argv[1]):
  splits = line.strip().split()
  if len(splits) != 2:
    sys.stderr.write("Cannot create feats.scp if it does not have ark files in it")
    sys.exit(0)
  feats[splits[0]] = splits[1]

start_times = dict()
end_times = dict()
for line in open(sys.argv[2]):
  splits = line.strip().split()
  start_times[splits[0]] = float(splits[2])
  end_times[splits[0]] = float(splits[3])

lengths = dict()
for line in open(sys.argv[3]):
  splits = line.strip().split()
  lengths[splits[0]] = int(splits[1])

frame_shift = float(sys.argv[4])

for line in sys.stdin.readlines():
  splits = line.strip().split()
  beg = int(float(splits[2]) / frame_shift * 1000.0)
  end = int((float(splits[3]) + start_times[splits[1]]) / frame_shift * 1000.0 + 0.5)
  end = min(end, lengths[splits[1]]) - 1
  print ("%s %s[%d:%d]" % (splits[0], feats[splits[1]], beg, end))' $srcdir/feats.scp $srcdir/segments $destdir/len $frame_shift < $sub_segments > \
    $destdir/feats.scp || exit 1
    
    if [ -f $srcdir/cmvn.scp ]; then
      cp $srcdir/cmvn.scp $destdir
    fi
  fi

  if [ -f $srcdir/reco2file_and_channel ]; then
    cp $srcdir/reco2file_and_channel $destdir/
  fi
else
  echo "resegment_data_dir.sh: no such file $srcdir/segments"
  exit 1;
fi

if [ -f $sub_segment_text ]; then
  cp $sub_segment_text $destdir/text
else
  echo "$0: no such file $sub_segment_text"
  exit 1;
fi

if [ -f $srcdir/spk2gender ]; then
  cp $srcdir/spk2gender $destdir
fi

utils/utt2spk_to_spk2utt.pl $destdir/utt2spk > $destdir/spk2utt

for f in stm glm ctm; do
  if [ -f $srcdir/$f ]; then
    cp $srcdir/$f $destdir
  fi
done

rm $destdir/utt2dur 2>/dev/null || true
utils/data/get_utt2dur.sh $destdir

rm $destdir/utt_new2utt_map $destdir/utt2reco_map

echo "$0: resegmented data from $srcdir to $destdir"

[ ! -f $srcdir/feats.scp ] && validate_opts="$validate_opts --no-feats"
[ ! -f $srcdir/text ] && validate_opts="$validate_opts --no-text"

utils/validate_data_dir.sh $validate_opts $destdir
