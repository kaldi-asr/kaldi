#!/bin/bash

# Copyright    2013  Daniel Povey
#              2014  David Snyder
# Apache 2.0.

# This script gets gender-id information for a set of utterances.
# The output is a file utt2gender in the experimental directory.

# Begin configuration section.
nj=10
cmd="run.pl"
stage=-4
num_gselect1=20 # Gaussian-selection using diagonal model: number of Gaussians to select
num_gselect2=3 # Gaussian-selection using full-covariance model.
male_prior=0.5
cleanup=true
# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;


if [ $# != 5 ]; then
  echo "Usage: $0 <gender-independent-ubm-dir> <male-ubm-dir> <female-ubm-dir> <data> <exp-dir>"
  echo " e.g.: $0  exp/ubm_2048 exp/ubm_2048_male exp/ubm_2048_female data/test exp/test_gender"
  echo "main options (for others, see top of script file)"
  echo "  --config <config-file>                           # config containing options"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --nj <n|10>                                      # Number of jobs (also see num-processes and num-threads)"
  echo "  --male-prior <p|0.5>                             # Prior probability of male speaker"
  echo "  --cleanup <true,false|true>                      # If true, clean up temporary files"
  echo "  --num-processes <n|4>                            # Number of processes for each queue job (relates"
  echo "                                                   # to summing accs in memory)"
  echo "  --num-threads <n|4>                              # Number of threads for each process (can't be usefully"
  echo "                                                   # increased much above 4)"
  echo "  --stage <stage|-4>                               # To control partial reruns"
  echo "  --num-gselect <n|20>                             # Number of Gaussians to select using"
  echo "                                                   # diagonal model."
  echo "  --sum-accs-opt <option|''>                       # Option e.g. '-l hostname=a15' to localize"
  echo "                                                   # sum-accs process to nfs server."
  exit 1;
fi

ubmdir=$1
male_ubmdir=$2
female_ubmdir=$3
data=$4
dir=$5

delta_opts=`cat $ubmdir/delta_opts 2>/dev/null`
if [ -f $ubmdir/delta_opts ]; then
  cp $ubmdir/delta_opts $male_ubmdir/ 2>/dev/null
  cp $ubmdir/delta_opts $female_ubmdir/ 2>/dev/null
fi

for f in $ubmdir/final.ubm $male_ubmdir/final.ubm $female_ubmdir/final.ubm $data/feats.scp $data/vad.scp; do
  [ ! -f $f ] && echo "No such file $f" && exit 1;
done

# Set various variables.
mkdir -p $dir/log || exit 1;
sdata=$data/split$nj
utils/split_data.sh $data $nj || exit 1;

ng1=$(fgmm-global-info --print-args=false $ubmdir/final.ubm | grep gaussians | awk '{print $NF}')
ng2=$(fgmm-global-info --print-args=false $male_ubmdir/final.ubm | grep gaussians | awk '{print $NF}')
ng3=$(fgmm-global-info --print-args=false $female_ubmdir/final.ubm | grep gaussians | awk '{print $NF}')
if ! [ $ng1 -eq $ng2 ] || ! [ $ng1 -eq $ng3 ]; then
  echo "$0:  Number of Gaussians mismatch between speaker-independent, male "
  echo "$0:  and female UBMs: $ng1 vs $ng2 vs $ng3"
  exit 1;
fi


## Set up features.
feats="ark,s,cs:add-deltas $delta_opts scp:$sdata/JOB/feats.scp ark:- | apply-cmvn-sliding --norm-vars=false --center=true --cmn-window=300 ark:- ark:- | select-voiced-frames ark:- scp,s,cs:$sdata/JOB/vad.scp ark:- |"

if [ $stage -le -2 ]; then
  $cmd $dir/log/convert.log \
    fgmm-global-to-gmm $ubmdir/final.ubm $dir/final.dubm || exit 1;
fi

# Do Gaussian selection using diagonal form of model and then the full-covariance model.
# Even though this leads to, in some sense, less accurate likelihoods, I think it
# may improve the results for the same reason it sometimes helps to used fixed
# Gaussian posteriors rather than posteriors from the adapted model.

if [ $stage -le -1 ]; then
  echo $nj > $dir/num_jobs
  echo "$0: doing Gaussian selection"
  $cmd JOB=1:$nj $dir/log/gselect.JOB.log \
    gmm-gselect --n=$num_gselect1 $dir/final.dubm "$feats" ark:- \| \
    fgmm-gselect --gselect=ark,s,cs:- --n=$num_gselect2 $ubmdir/final.ubm \
      "$feats" "ark:|gzip -c >$dir/gselect.JOB.gz" || exit 1;
fi

if ! [ $nj -eq $(cat $dir/num_jobs) ]; then
  echo "Number of jobs mismatch" 
  exit 1;
fi


if [ $stage -le 0 ]; then
  $cmd JOB=1:$nj $dir/log/get_male_logprob.JOB.log \
    fgmm-global-get-frame-likes --average=true \
     "--gselect=ark,s,cs:gunzip -c $dir/gselect.JOB.gz|" $male_ubmdir/final.ubm \
      "$feats" ark,t:$dir/male_logprob.JOB || exit 1;
fi
if [ $stage -le 1 ]; then
  $cmd JOB=1:$nj $dir/log/get_female_logprob.JOB.log \
    fgmm-global-get-frame-likes --average=true \
     "--gselect=ark,s,cs:gunzip -c $dir/gselect.JOB.gz|" $female_ubmdir/final.ubm \
      "$feats" ark,t:$dir/female_logprob.JOB || exit 1;
fi

if [ $stage -le 2 ]; then

  for j in $(seq $nj); do cat $dir/male_logprob.$j; done > $dir/male_logprob
  for j in $(seq $nj); do cat $dir/female_logprob.$j; done > $dir/female_logprob

  n1=$(cat $dir/male_logprob | wc -l)
  n2=$(cat $dir/female_logprob | wc -l)

  if [ $n1 -ne $n2 ]; then
    echo "Number of lines mismatch, male versus female UBM probs: $n1 vs $n2"
    exit 1;
  fi

  paste $dir/male_logprob $dir/female_logprob | \
    awk '{if ($1 != $3) { print >/dev/stderr "Sorting mismatch"; exit(1);  } print $1, $2, $4;}' \
    >$dir/logprob || exit 1;

  cat $dir/logprob | \
    awk -v pmale=$male_prior '{lratio = log(pmale/(1-pmale))+$2-$3; print $1, 1/(1+exp(-lratio));}' \
    >$dir/ratio || exit 1;

  cat $dir/ratio | awk '{if ($2 > 0.5) { print $1, "m"; } else { print $1, "f"; }}' > $dir/utt2gender
fi

if [ $stage -le 3 ] && [ -f $data/spk2gender ]; then
  utils/apply_map.pl -f 2 $data/spk2gender  <$data/utt2spk | \
    utils/filter_scp.pl $dir/utt2gender > $dir/utt2gender.ref
  n1=$(cat $dir/utt2gender | wc -l)
  n2=$(cat $dir/utt2gender.ref | wc -l)
  ! [ $n1 -eq $n2 ] && echo "Number-of-utterances mismatch $n1 vs $n2" && exit 1;
  ! paste $dir/utt2gender $dir/utt2gender.ref | awk '{if ($1 != $3) { exit(1); }}' && \
     echo "sorting problem, compare $dir/utt2gender and $dir/utt2gender.ref" && exit 1;
  ! paste $dir/utt2gender $dir/utt2gender.ref | awk '{if ($2 != $4) { print; }}' > $dir/utt2gender.incorrect
  n3=$(cat $dir/utt2gender.incorrect | wc -l)
  
  err=$(perl -e "printf('%.2f', (100.0 * $n3 / $n1));")
  echo "Gender-id error rate is $err%" | tee $dir/error_rate
fi


if $cleanup; then
  rm $dir/gselect.*.gz
fi

exit 0;

