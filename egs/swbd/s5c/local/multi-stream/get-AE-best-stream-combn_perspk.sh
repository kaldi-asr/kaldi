#!/bin/bash 

# Copyright 2012-2015 Brno University of Technology (author: Karel Vesely)
# Apache 2.0
# To be run from .. (one directory up from here)
# see ../run.sh for example

# Begin configuration section.
cmd=run.pl

## DNN opts
nnet_forward_opts=
use_gpu=no
htk_save=false
ivector=            # rx-specifier with i-vectors (ark-with-vectors),

# AANN opts
only_mse=true

# combn score opts
alpha=1.0
topN=-1

## Other stuff
tmpdir=           # non-default location of tmp_dir, 
                   # if not specified, uses /tmp/kaldi.XXXX

# End configuration section.


echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

set -euo pipefail

if [ $# != 6 ]; then
   echo "usage: $0 [options] <src-data-dir> <strm-indices> <feat-transf-dir> <aann-dir> <log-dir> <out-dir>";
   echo "options: "
   echo "  --cmd 'queue.pl <queue opts>'   # how to run jobs."
   echo "  --nj <nj>                       # number of parallel jobs"
   echo "  --remove-last-components <N>    # number of NNet Components to remove from the end"
   echo "  --use-gpu (no|yes|optional)     # forwarding on GPU"
   exit 1;
fi

srcdata=$1
strm_indices=$2
transf_dir=$3 # nnet_feats + transf_of_nnet_feats
aann_dir=$4
logdir=$5
outdir=$6

outdir=`perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir; ' $outdir ${PWD}`

# Check required 
required="$srcdata/feats.scp"
required=$required" $transf_dir/feature_extractor.nnet $transf_dir/final.feature_transform $transf_dir/pca.mat"
required=$required" $aann_dir/final.feature_transform $aann_dir/final.nnet"
for f in $required; do
  [ ! -f $f ] && echo "$0: Missing $f" && exit 1;
done

num_streams=`echo $strm_indices | awk -F ":" '{print NF-1}'`

[ -z "$tmpdir" ] && tmpdir=$(mktemp -d /tmp/kaldi.XXXXXX)
trap "echo \"# Removing features $tmpdir @ $(hostname)\"; ls $tmpdir; rm -r $tmpdir" EXIT
mkdir -p $logdir
mkdir -p $outdir 

name=$(basename $srcdata)

feature_transform=$transf_dir/final.feature_transform
nnet=$transf_dir/feature_extractor.nnet
pcamat=$transf_dir/pca.mat

# PREPARE FEATURE EXTRACTION PIPELINE
# import config,
cmvn_opts=
delta_opts=
D=$transf_dir
[ -e $D/norm_vars ] && cmvn_opts="--norm-means=true --norm-vars=$(cat $D/norm_vars)" # Bwd-compatibility,
[ -e $D/cmvn_opts ] && cmvn_opts=$(cat $D/cmvn_opts)
[ -e $D/delta_order ] && delta_opts="--delta-order=$(cat $D/delta_order)" # Bwd-compatibility,
[ -e $D/delta_opts ] && delta_opts=$(cat $D/delta_opts)
#
# Create the feature stream,
feats="ark,s,cs:copy-feats scp:$srcdata/feats.scp ark:- |"
# apply-cmvn (optional),
[ ! -z "$cmvn_opts" -a ! -f $srcdata/cmvn.scp ] && echo "$0: Missing $srcdata/cmvn.scp" && exit 1
[ ! -z "$cmvn_opts" ] && feats="$feats apply-cmvn $cmvn_opts --utt2spk=ark:$srcdata/utt2spk scp:$srcdata/cmvn.scp ark:- ark:- |"
# add-deltas (optional),
[ ! -z "$delta_opts" ] && feats="$feats add-deltas $delta_opts ark:- ark:- |"
# add-pytel transform (optional),
[ -e $D/pytel_transform.py ] && feats="$feats /bin/env python $D/pytel_transform.py |"

# transf_nnet_out_opts
transf_nnet_out_opts=$(cat $transf_dir/transf_nnet_out_opts)
echo "Loading transf_nnet_out_opts=$transf_nnet_out_opts"

## create shell script
## that can be called from python  
echo "#!/bin/bash
cd $PWD
. ./path.sh

this_stream_combn=\$1
  
this_tmpdir=$tmpdir/comb.\${this_stream_combn}; mkdir -p \$this_tmpdir

multi_stream_opts=\"--cross-validate=true --stream-combination=\$this_stream_combn $strm_indices\"
multi_stream_feats=\"$feats nnet-forward $feature_transform ark:- ark:- | apply-feature-stream-mask \$multi_stream_opts ark:- ark:- |\"

nnet-forward $nnet_forward_opts --use-gpu=$use_gpu $nnet \"\$multi_stream_feats\" ark:- | \
  transform-nnet-posteriors $transf_nnet_out_opts ark:- ark:- | transform-feats ${pcamat} ark:- \
  ark,scp:\$this_tmpdir/pca_transf_$name.ark,\$this_tmpdir/pca_transf_$name.scp && \
  compute-cmvn-stats --spk2utt=ark:${srcdata}/spk2utt scp:\$this_tmpdir/pca_transf_$name.scp \
  ark,scp:\$this_tmpdir/pca_transf_$name.cmvn.ark,\$this_tmpdir/pca_transf_$name.cmvn.scp && \
  nnet-score-sent --use-gpu=no --objective-function=mse --feature-transform=${aann_dir}/final.feature_transform \
  ${aann_dir}/final.nnet \"ark,s,cs:copy-feats scp:\$this_tmpdir/pca_transf_$name.scp ark:- | apply-cmvn --norm-means=true --norm-vars=true --utt2spk=ark:${srcdata}/utt2spk scp:\$this_tmpdir/pca_transf_$name.cmvn.scp ark:- ark:- |\" \"ark,s,cs:copy-feats scp:\$this_tmpdir/pca_transf_$name.scp ark:- | apply-cmvn --norm-means=true --norm-vars=true --utt2spk=ark:${srcdata}/utt2spk scp:\$this_tmpdir/pca_transf_$name.cmvn.scp ark:- ark:- | nnet-forward ${aann_dir}/final.feature_transform ark:- ark:- | feat-to-post ark:- ark:- |\" ark:- | \
  select-feats 0 ark:- ark,scp:\$this_tmpdir/mse.ark,\$this_tmpdir/mse.scp && \
  python utils/multi-stream/pm_utils/mse_to_pkl_scores.py \$this_tmpdir/mse.scp $outdir/comb.\${this_stream_combn}.pklz || exit 1;

  rm -rf \$this_tmpdir/
" >$outdir/compute_stream_combn_scores.task


# gen utt_list
feat-to-len scp:$srcdata/feats.scp ark,t:- >$tmpdir/feat_to_len_list

### find best combination
## call this python function
python local/multi-stream/get-AE-best-stream-combn_perspk.py --alpha=${alpha} --topN=${topN} $outdir $num_streams $tmpdir/feat_to_len_list $srcdata/spk2utt $outdir/compute_stream_combn_scores.task $outdir/ | copy-feats ark:- ark,scp:$outdir/strm_mask.ark,$outdir/strm_mask.scp || exit 1;

exit 0;

