#!/bin/bash 

# Copyright 2012-2013  Johns Hopkins University (Author: Daniel Povey)
#                      Bagher BabaAli
# Apache 2.0
# To be run from .. (one directory up from here)
# This makes seven-dimension fundamental frequency variation features for some data/ directory.

# Begin configuration section.
nj=4
cmd=run.pl
stage=0
frame_len=0.01
sample_freq=8000
cleanup=true
# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 3 ]; then
   echo "Usage: make_ffv.sh [options] <data-dir> <exp-dir> <path-to-ffvdir>";
   echo "Makes seven dimensional features, based on fundumental frequency"
   echo "variation extractor."
   echo "E.g.: make_ffv.sh data/train_ffv exp/make_ffv_train plp/"
   echo "Options: "
   echo "  --frame_len 0.01                                 # frame length "
   echo "  --sample_freq 8000                               # sampling frequency "
   echo "  --nj <nj>                                        # number of parallel jobs"
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   exit 1;
fi

set -e 
set -o pipefail

data=$1
expdir=$2
ffvdir=$3

# make $ffvdir an absolute pathname.
ffvdir=`perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir; ' $ffvdir ${PWD}`
# make $expdir an absolute pathname.
expdir=`perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir; ' $expdir ${PWD}`

# use "name" as part of name of the archive.
name=`basename $data`

mkdir -p $ffvdir || exit 1;
mkdir -p $expdir/log || exit 1;

scp=$data/wav.scp

[ -s $KALDI_ROOT ] || KALDI_ROOT=../../.. 

ffv_pkg_dir=$KALDI_ROOT/tools/pitch_trackers/ffv-1.0.1
# make $ffv_pkg_dir an absolute pathname.
ffv_pkg_dir=`perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:)\
 { $dir = "$pwd/$dir"; } print $dir; ' $ffv_pkg_dir ${PWD}`

ffv_script=$expdir/run_ffv.sh

if [ ! -f $ffv_pkg_dir/ffv ]; then
  echo "*Expecting the file $KALDI_ROOT/tools/pitch_trackers/ffv-1.0.1/ffv to exist"
  echo "*cd to $KALDI_ROOT/tools/, and run extras/install_ffv.sh"
  exit 1;
fi

wavdir=$ffvdir/temp_wav_$name
mkdir -p $wavdir

split_wavs=" "
for ((n=1; n<=nj; n++)); do
  split_wavs="$split_wavs $expdir/wav.$n.scp"
done

utils/split_scp.pl  $data/wav.scp $split_wavs

if [ -f $data/segments ] || grep '|' $data/wav.scp >/dev/null; then
  for ((n=1; n<=nj; n++)); do
    wav_scp=$expdir/split_wavs.$n.scp
    
    if [ -f $data/segments ]; then
      grep -F -f <(cat $expdir/wav.$n.scp | awk '{print " "$1" ";}' ) \
        $data/segments > $expdir/segments.$n
    else
      # create a fake segments file that takes the whole file; this is an easy way
      # to copy to static wav files.  Note: probably this has not been tested.
      cat $splitdir/wav.scp | awk '{print $1, $1, 0.0, -1.0}' > $expdir/segments.$n
    fi

    cat $expdir/segments.$n | awk -v dir=$wavdir \
      '{key=$1; printf("%s %s/%s.wav\n", key, dir, key);}' \
      > $wav_scp || exit 1;
  done
else
  echo "No segments file exists, and wav scp is plain: using wav files as input."
  wav_scp=$splitdir/wav.scp
fi


segments=$expdir/segments
wav_scp=$expdir/wav.scp


if [ $stage -le 0 ]; then
  echo "Extracting wav-file segments (or just converting to wav format)"
  cat $expdir/segments.* > $segments
  cat $expdir/split_wavs.*.scp > $wav_scp
  
  extract-segments scp:$data/wav.scp $segments scp:$wav_scp || exit 1;

  wav_checked_scp=$expdir/wav_checked.scp
  cat $wav_scp | \
    perl -ane '@A=split; if (-f $A[1]) { print; } else {print STDERR ;}' >$wav_checked_scp 2>$expdir/missing_wavs.scp
  nl_orig=`cat $wav_scp | wc -l`
  nl_new=`cat $wav_checked_scp | wc -l`
  [ $nl_new -eq 0 ] && exit 1;
  echo "After removing non-existent files, number of utterances decreased from $nl_orig to $nl_new";
  segment_files=" "
  for ((n=1; n<=nj; n++)); do
    wav_scp_n=$expdir/split_wavs.$n.scp
    seg_scp=$expdir/segments.$n
    seg_scp_orig=$expdir/segments_orig.$n

    cp $seg_scp $seg_scp_orig
    cat $seg_scp_orig | grep -v -F -f <(cut -f 1 -d ' ' $expdir/missing_wavs.scp) > $seg_scp
    segment_files=" $segment_files $seg_scp"
    
    cat $expdir/segments.$n | awk -v dir=$wavdir \
      '{key=$1; printf("%s %s/%s.wav\n", key, dir, key);}' \
      > $wav_scp_n || exit 1;
  done
  cp $expdir/segments $expdir/segments_orig
  cat $segment_files | sort > $expdir/segments
  cat $expdir/split_wavs.*.scp > $wav_scp
fi

# For each wav file, create corresponding temporary ffv file, in the
# format the ffv outputs: [ffv[0] ffv[1] ... ffv[6]]
temp_ffvdir=$ffvdir/temp_ffv_$name
mkdir -p $temp_ffvdir

for ((n=1; n<=nj; n++)); do
  mkdir -p $temp_ffvdir/$n
  cat $expdir/split_wavs.$n.scp | awk -v pdir=$temp_ffvdir -v n=$n \
     '{key=$1; wavfile=$2; printf("%s,%s/%s/%s.ffv\n", wavfile, pdir, n, key);}' \
    > $expdir/ffv_flist.$n || exit 1
done

cat <<'EOF' > $ffv_script
#!/bin/bash
# script for execution of ffv
flen=0.01
sfreq=8000
. parse_options.sh || exit 1;
set -e
set -o pipefail

flist=$1
ffv_pkg_dir=$2
[ $# -ne 2 ] && echo "Usage: ffv.sh <ffv-flist-in> <ffv_pkg_dir>" && exit 1;
echo $flist
echo start running ffv
for wavefile in `cat $flist`; do
  echo wavefile : $wavefile 
  input=`echo $wavefile | cut -f1 -d ','`
  output=`echo $wavefile | cut -f2 -d ','`
  echo "input : $input"
  echo "output : $output"
  echo "Running: $ffv_pkg_dir/ffv --tfra $flen --fs $sfreq $input $output"
  [ -f $output ] && rm -r $output
  $ffv_pkg_dir/ffv --tfra $flen --fs $sfreq $input $output
done
EOF
chmod +x $ffv_script

if [ $stage -le 1 ]; then
  # Need to do this in director $ffv_pkg_dir as some of the things in its config
  # are relative pathnames.
  $cmd JOB=1:$nj $d/$expdir/log/ffv.JOB.log \
    $ffv_script --flen $frame_len --sfreq $sample_freq \
    $expdir/ffv_flist.JOB $ffv_pkg_dir || exit 1;
fi

# I don't want to put a separate script in svn just for this, so creating a temporary
# script file in the experimental directory.  Quotes around 'EOF' disable any 
# interpretation in the here-doc.
cat <<'EOF' > $expdir/convert.sh
#!/bin/bash
ffv_flist=$1 
scpfile=$2
[ $# -ne 2 ] && echo "Usage: convert.sh <ffv-flist-in> <scpfile-out>" && exit 1;

for f in `cat $ffv_flist | cut -d, -f2`; do
  g=`echo $f | sed s:.ffv$:.mat:`
  if [ -f $f ]; then
    cat $f | sed s:-nan:0:g | awk 'BEGIN{printf("[ "); } {print $1, $2, $3, $4, $5, $6, $7;} END{ print "]"; }' > $g
  fi
done
cat $ffv_flist | cut -d, -f2 | \
   perl -ane 'm:/([^/]+)\.ffv$: || die "Bad line $_"; $key=$1; s/\.ffv$/\.mat/; print "$key $_";' > $scpfile
EOF
chmod +x $expdir/convert.sh

if [ $stage -le 2 ]; then
  echo "Converting format from .ffv to .mat (kaldi-readable format)"
  $cmd JOB=1:$nj $expdir/log/convert.JOB.log \
    $expdir/convert.sh $expdir/ffv_flist.JOB $expdir/mat.scp.JOB || exit 1;
fi
if [ $stage -le 3 ]; then
  echo "Doing final processing (interpolation, smoothing, etc.) on pitch features"
  $cmd JOB=1:$nj $expdir/log/process.JOB.log \
    copy-matrix scp,p:$expdir/mat.scp.JOB  \
      ark,scp:$ffvdir/${name}_ffv.JOB.ark,$ffvdir/${name}_ffv.JOB.scp || exit 1;
fi

sleep 10s

echo "Creating $data/feats.scp"
for ((n=1; n<=nj; n++)); do cat $ffvdir/${name}_ffv.$n.scp; done > $data/feats.scp

if $cleanup; then
  echo "Removing temporary files"
  rm -r $wavdir $temp_ffvdir
fi

utils/summarize_logs.pl $expdir/log

echo "Finished extracting ffv features for $name"

exit 0;
