#!/usr/bin/env bash

# Copyright 2012-2013  Johns Hopkins University (Author: Daniel Povey)
#                      Bagher BabaAli
# Apache 2.0
# To be run from .. (one directory up from here)
# This makes two-dimension p(voicing) and pitch features for some data/ directory.

# Begin configuration section.
nj=4
cmd=run.pl
stage=0
pitch_config=
interpolate_pitch_opts=
process_pitch_opts=
cleanup=true
# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 3 ]; then
   echo "Usage: make_pitch.sh [options] <data-dir> <exp-dir> <path-to-pitchdir>";
   echo "Makes two dimensional [p(voicing), pitch] features, based on SAcC pitch"
   echo "extractor followed by some normalization and smoothing"
   echo "E.g.: make_pitch.sh data/train_pitch exp/make_pitch_train plp/"
   echo "Options: "
   echo "  --pitch-config <config-file>                     # config passed to compute-pitch-feats "
   echo "  --nj <nj>                                        # number of parallel jobs"
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   exit 1;
fi

data=$1
expdir=$2
pitchdir=$3

# make $pitchdir an absolute pathname.
pitchdir=`perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir; ' $pitchdir ${PWD}`
# make $expdir an absolute pathname.
expdir=`perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir; ' $expdir ${PWD}`

# use "name" as part of name of the archive.
name=`basename $data`

mkdir -p $pitchdir || exit 1;
mkdir -p $expdir/log || exit 1;

scp=$data/wav.scp

[ ! -s $KALDI_ROOT ] && KALDI_ROOT=../../..

( # this is for back compatiblity:
 cd $KALDI_ROOT/tools
 if [ -d sacc ] && [ ! -d pitch_trackers/sacc ]; then
   echo "Linking sacc directory to new location."
   mkdir -p pitch_trackers
   cd pitch_trackers
   ln -s ../sacc ..
 fi
)

sacc_dir=$KALDI_ROOT/tools/pitch_trackers/sacc/SAcC_GLNXA64/
# make $sacc_dir an absolute pathname.
sacc_dir=`perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir; ' $sacc_dir ${PWD}`

sacc_script=$sacc_dir/run_SAcC.sh
sacc_config=$sacc_dir/conf/Babelnet_sr8k_bpo6_sb24_k10.config

if [ ! -f $sacc_script ]; then
  echo "*Expecting the script $sacc_script to exist"
  echo "*cd to $KALDI_ROOT/tools/, and run extras/install_sacc.sh"
  echo "*Re-run this script when it is installed."
  exit 1;
fi

required="$scp $pitch_config $sacc_config"

for f in $required; do
  if [ ! -f $f ]; then
    echo "make_pitch.sh: no such file $f"
    exit 1;
  fi
done

# note: in general, the double-parenthesis construct in bash "((" is "C-style
# syntax" where we can get rid of the $ for variable names, and omit spaces.
# The "for" loop in this style is a special construct.

basename=`basename $data`
wavdir=$pitchdir/temp_wav_$basename
mkdir -p $wavdir

if [ -f $data/segments ] || grep '|' $data/wav.scp >/dev/null; then
  wav_scp=$expdir/wav.scp
  cat $data/segments | awk -v dir=$wavdir '{key=$1; printf("%s %s/%s.wav\n", key, dir, key);}' \
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
    $cmd  $expdir/log/extract-segments.log \
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

# For each wav file, create corresponding temporary pitch file, in the
# format the SAcC outputs: [ 0 frame pitch p(voicing) ]
temp_pitchdir=$pitchdir/temp_pitch_$basename
mkdir -p $temp_pitchdir

for ((n=1; n<=nj; n++)); do
  mkdir -p $temp_pitchdir/$n
  cat $expdir/split_wavs.$n.scp | awk -v pdir=$temp_pitchdir -v n=$n \
     '{key=$1; wavfile=$2; printf("%s,%s/%s/%s.pitch\n", wavfile, pdir, n, key);}' \
    > $expdir/sacc_flist.$n || exit 1
done

if [ $stage -le 1 ]; then
  # Need to do this in director $sacc_dir as some of the things in its config
  # are relative pathnames.
  $cmd JOB=1:$nj $d/$expdir/log/sacc.JOB.log \
    cd $sacc_dir '&&' $sacc_script $expdir/sacc_flist.JOB $sacc_config || exit 1;
fi

# I don't want to put a separate script in svn just for this, so creating a temporary
# script file in the experimental directory.  Quotes around 'EOF' disable any
# interpretation in the here-doc.
cat <<'EOF' > $expdir/convert.sh
#!/usr/bin/env bash
sacc_flist=$1
scpfile=$2
[ $# -ne 2 ] && echo "Usage: convert.sh <sacc-flist-in> <scpfile-out>" && exit 1;

for f in `cat $sacc_flist | cut -d, -f2`; do
  g=`echo $f | sed s:.pitch$:.mat:`
  if [ -f $f ]; then
    cat $f | awk 'BEGIN{printf("[ "); } {print $4, $3;} END{ print "]"; }' > $g
    rm $f
  fi
done
cat $sacc_flist | cut -d, -f2 | \
   perl -ane 'm:/([^/]+)\.pitch$: || die "Bad line $_"; $key=$1; s/\.pitch$/\.mat/; print "$key $_";' > $scpfile
EOF
chmod +x $expdir/convert.sh

if [ $stage -le 2 ]; then
  echo "Converting format from .pitch to .mat (kaldi-readable format)"
  $cmd JOB=1:$nj $expdir/log/convert.JOB.log \
    $expdir/convert.sh $expdir/sacc_flist.JOB $expdir/mat.scp.JOB || exit 1;
fi

if [ $stage -le 3 ]; then
  echo "Doing final processing (interpolation, smoothing, etc.) on pitch features"
  $cmd JOB=1:$nj $expdir/log/process.JOB.log \
    interpolate-pitch $interpolate_pitch_opts scp:$expdir/mat.scp.JOB ark:- \| \
    process-pitch-feats $process_pitch_opts ark:- \
      ark,scp:$pitchdir/${basename}_pitch.JOB.ark,$pitchdir/${basename}_pitch.JOB.scp || exit 1;
fi

echo "Creating $data/feats.scp"
for ((n=1; n<=nj; n++)); do cat $pitchdir/${basename}_pitch.$n.scp; done > $data/feats.scp

if $cleanup; then
  echo "Removing temporary files"
  rm -r $wavdir $temp_pitchdir
fi

echo "Finished extracting pitch features for $basename"

debug=~/temp2.m
echo "A = [" > $debug
copy-feats scp:$data/feats.scp ark,t:- | grep -v ']' | grep -v '\[' | awk '{if (NF == 2) { print; }}' | head -n 200000 \
  >> $debug

cat <<'EOF' >>$debug
];
pov = A(:, 1);
pitch = A(:, 2);
subplot(2, 2, 1);
hist(pov, 30);
legend('pov')
subplot(2, 2, 2);
hist(pitch, 30);
legend('pitch')

len=size(pov, 1);
povD = pov(1:len-1) - pov(2:len);
subplot(2, 2, 3);
hist(povD, 30);
legend('delta-pov')

pitchD = pitch(1:len-1) - pitch(2:len);
pitchD = max(pitchD, -0.05);
pitchD = min(pitchD, 0.05);
subplot(2, 2, 4);
hist(pitchD, 50);
legend('delta-pitch');

print -deps 'C.eps'
EOF

exit 0;


# Here's

#copy-feats scp:plp/train_pitch_pitch.10.scp ark,t:- | grep -v ']' | grep -v '\[' | awk '{if (NF == 2) { print; }}' | head -n 200000 > ~/temp2.m

#
### data goes here.
#];



#   rm $expdir/.error 2>/dev/null

# #  for ((n=1; n<=nj; n++)); do
# #     mkdir -p "$expdir/$n"
# #  done

# #  $cmd JOB=1:$nj $expdir/make_pitch.JOB.log \
# #    extract-segments scp:$scp $expdir/segments.JOB ark:- \| \
# #    compute-pitch-feats --verbose=2 --config=$pitch_config ark:- \
# #      ark,scp:$pitchdir/raw_pitch_$name.JOB.ark,$pitchdir/raw_pitch_$name.JOB.scp \
# #      `pwd`/$expdir/JOB || exit 1;

#   $cmd JOB=1:$nj $expdir/make_pitch.JOB.log \
#     extract-segments scp:$scp $expdir/segments.JOB ark:- \| \
#     local/SAcC.sh $expdir/wav.JOB.scp $pitchdir $name.JOB ||  exit 1;

# else
#   echo "$0: [info]: no segments file exists: assuming wav.scp indexed by utterance."
#   split_scps=""
#   for ((n=1; n<=nj; n++)); do
#     split_scps="$split_scps $expdir/wav.$n.scp"
#   done

#   utils/split_scp.pl $scp $split_scps || exit 1;

# #  for ((n=1; n<=nj; n++)); do
# #     mkdir -p "$expdir/$n"
# #  done

# #  $cmd JOB=1:$nj $expdir/make_pitch.JOB.log \
# #    compute-pitch-feats  --verbose=2 --config=$pitch_config scp:$expdir/wav.JOB.scp \
# #      ark,scp:$pitchdir/raw_pitch_$name.JOB.ark,$pitchdir/raw_pitch_$name.JOB.scp \
# #      $expdir/JOB || exit 1;

#   pushd $sacc_dir
#   $cmd JOB=1:$nj $expdir/make_pitch.JOB.log \
#     cd $sacclocal/SAcC.sh $expdir/wav.JOB.scp $pitchdir $name.JOB ||  exit 1;
# fi


# if [ -f $expdir/.error.$name ]; then
#   echo "Error producing pitch features for $name:"
#   tail $expdir/make_pitch.*.log
#   exit 1;
# fi

# # concatenate the .scp files together.
# for ((n=1; n<=nj; n++)); do
#   cat $pitchdir/raw_pitch_$name.$n.scp >> $data/pitchs.scp || exit 1;
# done > $data/pitchs.scp

# rm $expdir/wav.*.scp  $expdir/segments.* 2>/dev/null

# nf=`cat $data/pitchs.scp | wc -l`
# nu=`cat $data/utt2spk | wc -l`
# if [ $nf -ne $nu ]; then
#   echo "It seems not all of the feature files were successfully ($nf != $nu);"
#   echo "consider using utils/fix_data_dir.sh $data"
# fi

# echo "Succeeded creating PITCH features for $name"
