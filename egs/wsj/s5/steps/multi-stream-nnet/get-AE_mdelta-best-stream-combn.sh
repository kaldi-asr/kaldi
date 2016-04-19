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

per_spk=true
# combn score opts
alpha=1.0
topN=-1

# 
pms_combn_weights="0.5:0.5"
pms_train_stats=

## Other stuff
num_threads=5     # parallel score computation
tmpdir=           # non-default location of tmp_dir, 
                   # if not specified, uses /tmp/kaldi.XXXX

# End configuration section.


echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

set -euo pipefail

if [ $# != 7 ]; then
   echo "usage: $0 [options] <src-data-dir> <strm-indices> <feat-transf-dir> <aann-dir> <mdelta-stats-dir> <log-dir> <out-dir>";
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
mdelta_stats_dir=$5
logdir=$6
outdir=$7

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
autoencoder_transf_nnet_out_opts=$(cat $transf_dir/transf_nnet_out_opts)
echo "Loading autoencoder_transf_nnet_out_opts=$autoencoder_transf_nnet_out_opts"

mdelta_transf_nnet_out_opts="--pdf-to-pseudo-phone=$mdelta_stats_dir/pdf_to_pseudo_phone.txt"

## create shell script
## that can be called from python  
echo "#!/bin/bash
cd $PWD
. ./path.sh

this_stream_combn=\$1
  
this_tmpdir=$tmpdir/comb.\${this_stream_combn}; mkdir -p \$this_tmpdir

multi_stream_opts=\"--cross-validate=true --stream-combination=\$this_stream_combn $strm_indices\"
multi_stream_feats=\"$feats nnet-forward $feature_transform ark:- ark:- | apply-feature-stream-mask \$multi_stream_opts ark:- ark:- |\"

  nnet-forward $nnet_forward_opts --use-gpu=$use_gpu $nnet \"\$multi_stream_feats\" ark,scp:\$this_tmpdir/state_post.ark,\$this_tmpdir/state_post.scp || exit 1;

  # pm1
  transform-nnet-posteriors $autoencoder_transf_nnet_out_opts scp:\$this_tmpdir/state_post.scp ark:- | transform-feats ${pcamat} ark:- \
  ark,scp:\$this_tmpdir/pca_transf_$name.ark,\$this_tmpdir/pca_transf_$name.scp || exit 1;

  compute-cmvn-stats --spk2utt=ark:${srcdata}/spk2utt scp:\$this_tmpdir/pca_transf_$name.scp \
  ark,scp:\$this_tmpdir/pca_transf_$name.cmvn.ark,\$this_tmpdir/pca_transf_$name.cmvn.scp && \
  nnet-score-sent --use-gpu=no --objective-function=mse --feature-transform=${aann_dir}/final.feature_transform \
  ${aann_dir}/final.nnet \"ark,s,cs:copy-feats scp:\$this_tmpdir/pca_transf_$name.scp ark:- | apply-cmvn --norm-means=true --norm-vars=true --utt2spk=ark:${srcdata}/utt2spk scp:\$this_tmpdir/pca_transf_$name.cmvn.scp ark:- ark:- |\" \"ark,s,cs:copy-feats scp:\$this_tmpdir/pca_transf_$name.scp ark:- | apply-cmvn --norm-means=true --norm-vars=true --utt2spk=ark:${srcdata}/utt2spk scp:\$this_tmpdir/pca_transf_$name.cmvn.scp ark:- ark:- | nnet-forward ${aann_dir}/final.feature_transform ark:- ark:- | feat-to-post ark:- ark:- |\" ark:- | \
  select-feats 0 ark:- ark,scp:\$this_tmpdir/mse.ark,\$this_tmpdir/mse.scp || exit 1;
  python utils/multi-stream/pm_utils/mse_to_pkl_scores.py \$this_tmpdir/mse.scp $outdir/autoencoder.comb.\${this_stream_combn}.pklz || exit 1;

  (cd $outdir/; ln -s autoencoder.comb.\${this_stream_combn}.pklz pm1.comb.\${this_stream_combn}.pklz; cd -)

  # pm2
  transform-nnet-posteriors $mdelta_transf_nnet_out_opts scp:\$this_tmpdir/state_post.scp ark,scp:\$this_tmpdir/post.ark,\$this_tmpdir/post.scp || exit 1;
  python utils/multi-stream/pm_utils/compute_mdelta_post.py \$this_tmpdir/post.scp $mdelta_stats_dir/pri $outdir/mdelta.comb.\${this_stream_combn}.pklz || exit 1;
  (cd $outdir/; ln -s mdelta.comb.\${this_stream_combn}.pklz pm2.comb.\${this_stream_combn}.pklz; cd -)

  # combine them?
  python utils/multi-stream/pm_utils/combine_pm-scores.py --weights=${pms_combn_weights} --pm-stats-files=$pms_train_stats $outdir/pm1.comb.\${this_stream_combn}.pklz $outdir/pm2.comb.\${this_stream_combn}.pklz $outdir/comb.\${this_stream_combn}.pklz || exit 1; 

  rm -rf \$this_tmpdir/
" >$outdir/compute_stream_combn_scores.task


# gen utt_list
feat-to-len scp:$srcdata/feats.scp ark,t:- >$tmpdir/feat_to_len_list
# gen utt_list
feat-to-len scp:$srcdata/feats.scp ark,t:- >$tmpdir/feat_to_len_list
if [ $per_spk == "false" ]; then
  awk '{print $1, $1}' $srcdata/utt2spk > $tmpdir/spk2utt
else
  cp $srcdata/spk2utt $tmpdir/spk2utt
fi

### find best combination
## call this python function
python utils/multi-stream/pm_utils/get-best-stream-combn_perspk.py \
  --alpha=${alpha} --topN=${topN} --num-threads $num_threads \
  $outdir $num_streams $tmpdir/feat_to_len_list $srcdata/spk2utt $outdir/compute_stream_combn_scores.task $outdir/ | \
  copy-feats ark:- ark,scp:$outdir/strm_mask.ark,$outdir/strm_mask.scp || exit 1;

exit 0;

