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
ffv_config=
interpolate_ffv_opts=
process_ffv_opts=
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
   echo "  --ffv-config <config-file>                     # config passed to compute-ffv-feats "
   echo "  --nj <nj>                                        # number of parallel jobs"
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   exit 1;
fi

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

[ -s $KALDI_ROOT ] && KALDI_ROOT=../../.. 

ffv_pkg_dir=$KALDI_ROOT/tools/pitch_trackers/ffv-1.0.1
# make $ffv_pkg_dir an absolute pathname.
ffv_pkg_dir=`perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:)\
 { $dir = "$pwd/$dir"; } print $dir; ' $ffv_pkg_dir ${PWD}`

ffv_script=$ffv_pkg_dir/run_ffv.sh
ffv_config=$ffv_pkg_dir/conf/1.config

if [ ! -f $ffv_pkg_dir/ffv ]; then
  echo "*Expecting the file $KALDI_ROOT/tools/pitch_trackers/ffv-1.0.1/ffv to exist"
  echo "*cd to $KALDI_ROOT/tools/, and run extras/install_ffv.sh"
  exit 1;
fi
if false; then
required="$scp $ffv_config"

for f in $required; do
  if [ ! -f $f ]; then
    echo "make_ffv.sh: no such file $f"
    exit 1;
  fi
done
fi #100

# note: in general, the double-parenthesis construct in bash "((" is "C-style
# syntax" where we can get rid of the $ for variable names, and omit spaces.
# The "for" loop in this style is a special construct.

basename=`basename $data`
wavdir=$ffvdir/temp_wav_$basename
mkdir -p $wavdir

if [ -f $data/segments ] || grep '|' $data/wav.scp >/dev/null; then
  wav_scp=$expdir/wav.scp
  cat $data/segments | awk -v dir=$wavdir \
  '{key=$1; printf("%s %s/%s.wav\n", key, dir, key);}' \
    > $wav_scp || exit 1;

  if [ -f $data/segments ]; then
    echo "$0 [info]: segments file exists: creating temporary wav files in $wavdir"
    segments=$data/segments
  else
    # create a fake segments file that takes the whole file; this is an easy way
    # to copy to static wav files.  Note: probably this has not been tested.
    cat $data/wav.scp | awk '{print $1, $1, 0.0, -1.0}' > $expdir/fake_segments 
    segments=$expdir/fake_segments
  fi
  if [ $stage -le 0 ]; then
    echo "Extracting wav-file segments (or just converting to wav format)"
    extract-segments scp:$data/wav.scp $segments scp:$wav_scp || exit 1;
  fi
else
  echo "No segments file exists, and wav scp is plain: using wav files as input."
  wav_scp=$data/wav.scp
fi

wav_checked_scp=$expdir/wav_checked.scp
cat $wav_scp | \
  perl -ane '@A=split; if (-f $A[1]) { print; }' >$wav_checked_scp
nl_orig=`cat $wav_scp | wc -l`
nl_new=`cat $wav_checked_scp | wc -l`
echo "After removing non-existent files, number of utterances decreased from $nl_orig to $nl_new";
[ $nl_new -eq 0 ] && exit 1;

# now $wav_scp is an scp file for the per-utterance wav files.

# Split up the wav files into multiple lists.
split_wavs=""
for ((n=1; n<=nj; n++)); do
  split_wavs="$split_wavs $expdir/split_wavs.$n.scp"
done
utils/split_scp.pl $wav_checked_scp $split_wavs || exit 1;

# For each wav file, create corresponding temporary ffv file, in the
# format the ffv outputs: [ffv[0] ffv[1] ... ffv[6]]
temp_ffvdir=$ffvdir/temp_ffv_$basename
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
flist=$1
ffv_pkg_dir=$2
echo $flist
echo start running ffv
for wavefile in `cat $flist`; do
  echo wavefile : $wavefile 
  input=`echo $wavefile | cut -f1 -d ','`
  output=`echo $wavefile | cut -f2 -d ','`
  echo input : $input and output : $output
  if [ ! -f $output ]; then
    echo "no such file $output"  
  fi  
  if [ ! -f $3/$basename.out ]; then 
    $ffv_pkg_dir/ffv --tfra 0.01 --fs 8000 $input $output
  fi
done
EOF
chmod +x $ffv_script

if [ $stage -le 1 ]; then
  # Need to do this in director $ffv_pkg_dir as some of the things in its config
  # are relative pathnames.
  $cmd JOB=1:$nj $d/$expdir/log/ffv.JOB.log \
    $ffv_script $expdir/ffv_flist.JOB $ffv_pkg_dir || exit 1;
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
    cat $f | awk 'BEGIN{printf("[ "); } {print $1, $2, $3, $4, $5, $6, $7;} END{ print "]"; }' > $g
    rm $f
  fi
done
cat $ffv_flist | cut -d, -f2 | \
   perl -ane 'm:/([^/]+)\.ffv: || die "Bad line $_"; $key=$1; s/\.ffv$/\.mat/; print "$key $_";' > $scpfile
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
      ark,scp:$ffvdir/${basename}_ffv.JOB.ark,$ffvdir/${basename}_ffv.JOB.scp || exit 1;
fi

echo "Creating $data/feats.scp"
for ((n=1; n<=nj; n++)); do cat $ffvdir/${basename}_ffv.$n.scp; done > $data/feats.scp

if $cleanup; then
  echo "Removing temporary files"
  rm -r $wavdir $temp_ffvdir
fi

echo "Finished extracting ffv features for $basename"

exit 0;
