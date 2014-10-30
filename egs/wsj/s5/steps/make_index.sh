#!/bin/bash

# Copyright 2012  Johns Hopkins University (Author: Guoguo Chen)
# Apache 2.0

# Begin configuration section.  
model= # You can specify the model to use
cmd=run.pl
acwt=0.083333
lmwt=1.0
max_silence_frames=50
max_states=1000000
max_states_scale=4
max_expand=180 # limit memory blowup in lattice-align-words
strict=true
word_ins_penalty=0
silence_word=  # Specify this only if you did so in kws_setup
skip_optimization=false     # If you only search for few thousands of keywords, you probablly
                            # can skip the optimization; but if you're going to search for 
                            # millions of keywords, you'd better do set this optimization to 
                            # false and do the optimization on the final index.
# End configuration section.

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

if [ $# != 4 ]; then
   echo "Usage: steps/make_index.sh [options] <kws-data-dir> <lang-dir> <decode-dir> <kws-dir>"
   echo "... where <decode-dir> is where you have the lattices, and is assumed to be"
   echo " a sub-directory of the directory where the model is."
   echo "e.g.: steps/make_index.sh data/kws data/lang exp/sgmm2_5a_mmi/decode/ exp/sgmm2_5a_mmi/decode/kws/"
   echo ""
   echo "main options (for others, see top of script file)"
   echo "  --acwt <float>                                   # acoustic scale used for lattice"
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   echo "  --lmwt <float>                                   # lm scale used for lattice"
   echo "  --model <model>                                  # which model to use"
   echo "                                                   # speaker-adapted decoding"
   echo "  --max-silence-frames <int>                       # maximum #frames for silence"
   exit 1;
fi


kwsdatadir=$1;
langdir=$2;
decodedir=$3;
kwsdir=$4;
srcdir=`dirname $decodedir`; # The model directory is one level up from decoding directory.

mkdir -p $kwsdir/log;
nj=`cat $decodedir/num_jobs` || exit 1;
echo $nj > $kwsdir/num_jobs;
word_boundary=$langdir/phones/word_boundary.int
utter_id=$kwsdatadir/utter_id

if [ -z "$model" ]; then # if --model <mdl> was not specified on the command line...
  model=$srcdir/final.mdl; 
fi

for f in $word_boundary $model $decodedir/lat.1.gz; do
  [ ! -f $f ] && echo "make_index.sh: no such file $f" && exit 1;
done

echo "Using model: $model"

if [ ! -z $silence_word ]; then
  silence_int=`grep -w $silence_word $langdir/words.txt | awk '{print $2}'`
  [ -z $silence_int ] && \
    echo "Error: could not find integer representation of silence word $silence_word" && exit 1;
  silence_opt="--silence-label=$silence_int"
fi

$cmd JOB=1:$nj $kwsdir/log/index.JOB.log \
  lattice-add-penalty --word-ins-penalty=$word_ins_penalty "ark:gzip -cdf $decodedir/lat.JOB.gz|" ark:- \| \
    lattice-align-words $silence_opt --max-expand=$max_expand $word_boundary $model  ark:- ark:- \| \
    lattice-scale --acoustic-scale=$acwt --lm-scale=$lmwt ark:- ark:- \| \
    lattice-to-kws-index --max-states-scale=$max_states_scale --allow-partial=true \
    --max-silence-frames=$max_silence_frames --strict=$strict ark:$utter_id ark:- ark:- \| \
    kws-index-union --skip-optimization=$skip_optimization --strict=$strict --max-states=$max_states \
    ark:- "ark:|gzip -c > $kwsdir/index.JOB.gz" || exit 1
    

exit 0;
