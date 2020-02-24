#!/usr/bin/env bash

# Copyright    2015 David Snyder
# Apache 2.0
#
# Compute GMM-based VAD output and optionally combine with
# the energy-based VAD decisions.

nj=10
cmd=run.pl
map_config=
merge_map_config=
priors=
use_energy_vad=false
num_gselect=20
norm_vars=false
center=true
stage=-4

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# -lt 5 ]; then
   echo "Usage: $0 [options] <data-dir> <gmm-dir-1> ... <gmm-dir-N> <log-dir> <vad-dir>";
   echo "e.g.: $0 data/train exp/music_gmm exp/speech_gmm exp/noise_gmm exp/gmm_vad exp/gmm_vad"
   echo " Options:"
   echo "  --map-config <config-file>                       # config passed to compute-vad-from-frame-likes"
   echo "  --priors <comma-separated-floats>                # list passed to compute-vad-from-frame-likes"
   echo "  --merge-map-config <config-file>                 # config passed to merge-vads"
   echo "  --use-energy-vad <true,false>                    # If true, look for a vad.scp file and combine it with this VAD"
   echo "  --nj <nj>                                        # number of parallel jobs"
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   exit 1;
fi

args=("$@")
gmm_dirs=(${@:2:$(($#-3))}) # The GMM directories
num_gmms=`expr $# - 3`

data=${args[0]}
log_dir=${args[$num_gmms+1]}
vad_dir=${args[$num_gmms+2]}

# make $vad_dir an absolute pathname.
vad_dir=`perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir; ' ${vad_dir} ${PWD}`
# use "name" as part of name of the archive.
name=`basename $data`

mkdir -p $vad_dir || exit 1;
mkdir -p $log_dir || exit 1;

if $use_energy_vad; then
  for f in $data/vad.scp "$merge_map_config"; do
    if [ ! -f $f ]; then
      echo "compute_vad_decision_gmm.sh: no such file $f"
      exit 1;
    fi
  done
fi

if [ ! -f $data/feats.scp ]; then
  echo "compute_vad_decision_gmm.sh: no such file $f"
  exit 1;
fi

utils/split_data.sh $data $nj || exit 1;
sdata=$data/split$nj;

# We assume that the same delta-opts is used for each
# GMM dir.
delta_opts=`cat ${gmm_dirs[0]}/delta_opts 2>/dev/null`
if [ -f ${gmm_dirs[0]}/delta_opts ]; then
  cp ${gmm_dirs[0]}/delta_opts $dir/ 2>/dev/null
fi

## Set up features.
feats="ark,s,cs:add-deltas $delta_opts scp:$sdata/JOB/feats.scp ark:- | apply-cmvn-sliding --norm-vars=$norm_vars --center=$center --cmn-window=300 ark:- ark:- |"

if [ $stage -le -2 ]; then
  for gmm_dir in "${gmm_dirs[@]}";
  do
    gmm_name=`basename $gmm_dir`
    $cmd ${log_dir}/log/${gmm_name}_convert.log \
      fgmm-global-to-gmm ${gmm_dir}/final.ubm ${vad_dir}/${gmm_name}_final.dubm || exit 1;
  done
fi

if [ $stage -le -1 ]; then
  echo "$0: doing Gaussian selection"
  for gmm_dir in "${gmm_dirs[@]}";
  do
    gmm_name=`basename $gmm_dir`
    $cmd JOB=1:$nj ${log_dir}/log/${gmm_name}_gselect.JOB.log \
      gmm-gselect --n=$num_gselect ${vad_dir}/${gmm_name}_final.dubm "$feats" ark:- \| \
      fgmm-gselect --gselect=ark,s,cs:- --n=${num_gselect} ${gmm_dir}/final.ubm \
      "$feats" "ark:|gzip -c >${vad_dir}/${gmm_name}_gselect.JOB.gz" || exit 1;
  done
fi

frame_likes=""
if [ $stage -le 0 ]; then
  echo "$0: computing frame likelihoods"
  for gmm_dir in "${gmm_dirs[@]}";
  do
    gmm_name=`basename $gmm_dir`
    frame_likes="${frame_likes} ark:${vad_dir}/${gmm_name}_logprob.JOB.ark"
    $cmd JOB=1:$nj ${log_dir}/log/get_${gmm_name}_logprob.JOB.log \
      fgmm-global-get-frame-likes --average=false \
      "--gselect=ark,s,cs:gunzip -c ${vad_dir}/${gmm_name}_gselect.JOB.gz|" ${gmm_dir}/final.ubm \
      "$feats" ark:${vad_dir}/${gmm_name}_logprob.JOB.ark || exit 1;
  done

  echo "$0: computing VAD decisions from frame likelihoods"
  $cmd JOB=1:$nj ${log_dir}/log/make_vad_gmm_${name}.JOB.log \
    compute-vad-from-frame-likes --map=${map_config} --priors=$priors $frame_likes \
    ark,scp:${vad_dir}/vad_gmm_${name}.JOB.ark,${vad_dir}/vad_gmm_${name}.JOB.scp \
    || exit 1;

  if $use_energy_vad ; then
    echo "$0: merging with energy-based VAD decisions"
    $cmd JOB=1:$nj ${log_dir}/log/merge_vads_${name}.JOB.log \
      merge-vads --map=${merge_map_config} scp:$sdata/JOB/vad.scp \
      scp:${vad_dir}/vad_gmm_${name}.JOB.scp \
      ark,scp:${vad_dir}/vad_merged_${name}.JOB.ark,${vad_dir}/vad_merged_${name}.JOB.scp \
      || exit 1;
  fi

  echo "$0: moving old vad.scp to ${data}/vad.scp.bak"
  mv ${data}/vad.scp ${data}/vad.scp.bak

  for ((n=1; n<=nj; n++)); do
    if $use_energy_vad ; then
      cat ${vad_dir}/vad_merged_${name}.$n.scp || exit 1;
    else
      cat ${vad_dir}/vad_gmm_${name}.$n.scp || exit 1;
    fi
  done > ${data}/vad.scp
fi

nc=`cat $data/vad.scp | wc -l`
nu=`cat $data/feats.scp | wc -l`
if [ $nc -ne $nu ]; then
  echo "**Warning it seems not all of the speakers got VAD output ($nc != $nu);"
  echo "**validate_data_dir.sh will fail; you might want to use fix_data_dir.sh"
  [ $nc -eq 0 ] && exit 1;
fi

echo "$0 created GMM-based VAD output for $name"

if $cleanup ; then
  for gmm_dir in "${gmm_dirs[@]}";
  do
    gmm_name=`basename $gmm_dir`
    rm ${vad_dir}/${gmm_name}_gselect.*.gz
    rm ${vad_dir}/${gmm_name}_logprob.*.ark
  done
fi

exit 0;
