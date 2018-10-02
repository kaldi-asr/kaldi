#!/bin/bash

# Copyright 2014  Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0

# Begin configuration.
stage=0 # This allows restarting after partway, when something when wrong.
feature_type=mfcc
online_cmvn_config=conf/online_cmvn.conf
add_pitch=false
pitch_config=conf/pitch.conf
pitch_process_config=conf/pitch_process.conf
per_utt_basis=true # If true, then treat each utterance as a separate speaker
                   # for purposes of basis training... this is recommended if
                   # the number of actual speakers in your training set is less
                   # than (feature-dim) * (feature-dim+1).
per_utt_cmvn=false # If true, apply online CMVN normalization per utterance
                   # rather than per speaker.
silence_weight=0.01
cmd=run.pl
cleanup=true
# End configuration.

echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh;
. parse_options.sh || exit 1;

if [ $# -ne 4 -a $# -ne 5 ]; then
   echo "Usage: $0 [options] <data-dir> <lang-dir> <sat-model-dir> [<MMI-model>] <output-dir>"
   echo "e.g.: $0 data/train data/lang exp/tri3b exp/tri3b_mmi/final.mdl exp/tri3b_online"
   echo "main options (for others, see top of script file)"
   echo "  --feature-type <mfcc|plp>                        # Type of the base features; "
   echo "                                                   # important to generate the correct"
   echo "                                                   # configs in <output-dir>/conf/"
   echo "  --online-cmvn-config <config>                    # config for online cmvn,"
   echo "                                                   # default conf/online_cmvn.conf"
   echo "  --add-pitch <true|false>                         # Append pitch features to cmvn"
   echo "                                                   # (default: false)"
   echo "  --per-utt-cmvn <true|false>                      # Apply online CMVN per utt, not"
   echo "                                                   # per speaker (default: false)"
   echo "  --per-utt-basis <true|false>                     # Do basis computation per utterance"
   echo "                                                   # (default: true)"
   echo "  --silence-weight <weight>                        # Weight on silence for basis fMLLR;"
   echo "                                                   # default 0.01."
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   echo "  --config <config-file>                           # config containing options"
   echo "  --stage <stage>                                  # stage to do partial re-run from."
   exit 1;
fi


if [ $# -eq 5 ]; then
  data=$1
  lang=$2
  srcdir=$3
  mmi_model=$4
  dir=$5
else
  data=$1
  lang=$2
  srcdir=$3
  mmi_model=$srcdir/final.mdl
  dir=$4
fi


for f in $srcdir/final.mdl $srcdir/ali.1.gz $data/feats.scp $lang/phones.txt \
    $mmi_model $online_cmvn_config; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

nj=`cat $srcdir/num_jobs` || exit 1;
sdata=$data/split$nj;
[[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh $data $nj || exit 1;

mkdir -p $dir/log
echo $nj >$dir/num_jobs || exit 1;

utils/lang/check_phones_compatible.sh $lang/phones.txt $srcdir/phones.txt || exit 1;
cp $lang/phones.txt $dir || exit 1;

splice_opts=`cat $srcdir/splice_opts 2>/dev/null`
cmvn_opts=`cat $srcdir/cmvn_opts 2>/dev/null`
silphonelist=`cat $lang/phones/silence.csl` || exit 1;
cp $srcdir/splice_opts $srcdir/cmvn_opts $srcdir/final.mat $srcdir/final.mdl $dir/ 2>/dev/null

cp $mmi_model $dir/final.rescore_mdl

# Set up the unadapted features "$sifeats".
if [ -f $dir/final.mat ]; then feat_type=lda; else feat_type=delta; fi
if ! $per_utt_cmvn; then
  online_cmvn_spk2utt_opt=
else
  online_cmvn_spk2utt_opt="--spk2utt=ark:$sdata/JOB/spk2utt"
fi


# create global_cmvn.stats
if ! matrix-sum --binary=false scp:$data/cmvn.scp - >$dir/global_cmvn.stats 2>/dev/null; then
  echo "$0: Error summing cmvn stats"
  exit 1
fi

if $add_pitch; then
  skip_opt="--skip-dims=13:14:15" # should make this more general.
fi

echo "$0: feature type is $feat_type";
case $feat_type in
  delta) sifeats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | add-deltas ark:- ark:- |"
        online_sifeats="ark,s,cs:apply-cmvn-online $skip_opt --config=$online_cmvn_config $dir/global_cmvn.stats $online_cmvn_spk2utt_opt scp:$sdata/JOB/feats.scp ark:- | add-deltas ark:- ark:- |";;
  lda) sifeats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | splice-feats $splice_opts ark:- ark:- | transform-feats $dir/final.mat ark:- ark:- |"
       online_sifeats="ark,s,cs:apply-cmvn-online $skip_opt --config=$online_cmvn_config $online_cmvn_spk2utt_opt $dir/global_cmvn.stats scp:$sdata/JOB/feats.scp ark:- | splice-feats $splice_opts ark:- ark:- | transform-feats $dir/final.mat ark:- ark:- |";;
  *) echo "Invalid feature type $feat_type" && exit 1;
esac

# Set up the adapted features "$feats" for training set.
if [ -f $srcdir/trans.1 ]; then
  feats="$sifeats transform-feats --utt2spk=ark:$sdata/JOB/utt2spk ark:$srcdir/trans.JOB ark:- ark:- |";
else
  feats="$sifeats";
fi


if $per_utt_basis; then
  spk2utt_opt=  # treat each utterance as separate speaker when computing basis.
  echo "Doing per-utterance adaptation for purposes of computing the basis."
else
  echo "Doing per-speaker adaptation for purposes of computing the basis."
  [ `cat $sdata/spk2utt | wc -l` -lt $[41*40] ] && \
    echo "Warning: number of speakers is small, might be better to use --per-utt=true."
  spk2utt_opt="--spk2utt=ark:$sdata/JOB/spk2utt"
fi

if [ $stage -le 0 ]; then
  echo "$0: Accumulating statistics for basis-fMLLR computation"
# Note: we get Gaussian level alignments with the "final.mdl" and the
# speaker adapted features.
  $cmd JOB=1:$nj $dir/log/basis_acc.JOB.log \
    ali-to-post "ark:gunzip -c $srcdir/ali.JOB.gz|" ark:- \| \
    weight-silence-post $silence_weight $silphonelist $dir/final.mdl ark:- ark:- \| \
    gmm-post-to-gpost $dir/final.mdl "$feats" ark:- ark:- \| \
    gmm-basis-fmllr-accs-gpost $spk2utt_opt \
    $dir/final.mdl "$sifeats" ark,s,cs:- $dir/basis.acc.JOB || exit 1;
fi

if [ $stage -le 1 ]; then
  echo "$0: computing the basis matrices."
  $cmd $dir/log/basis_training.log \
    gmm-basis-fmllr-training $dir/final.mdl $dir/fmllr.basis $dir/basis.acc.* || exit 1;
  if $cleanup; then
    rm $dir/basis.acc.* 2>/dev/null
  fi
fi

if [ $stage -le 2 ]; then
  echo "$0: accumulating stats for online alignment model."

  # Accumulate stats for "online alignment model"-- this model is computed with
  # the speaker-independent features and online CMVN, but matches
  # Gaussian-for-Gaussian with the final speaker-adapted model.

  $cmd JOB=1:$nj $dir/log/acc_alimdl.JOB.log \
    ali-to-post "ark:gunzip -c $srcdir/ali.JOB.gz|" ark:-  \| \
    gmm-acc-stats-twofeats $dir/final.mdl "$feats" "$online_sifeats" \
    ark,s,cs:- $dir/final.JOB.acc || exit 1;
  [ `ls $dir/final.*.acc | wc -w` -ne "$nj" ] && echo "$0: Wrong #accs" && exit 1;
  # Update model.
  $cmd $dir/log/est_online_alimdl.log \
    gmm-est --remove-low-count-gaussians=false $dir/final.mdl \
    "gmm-sum-accs - $dir/final.*.acc|" $dir/final.oalimdl  || exit 1;
  if $cleanup; then
    rm $dir/final.*.acc
  fi
fi

if [ $stage -le 3 ]; then
  mkdir -p $dir/conf
  rm $dir/{plp,mfcc}.conf 2>/dev/null
  echo "$0: preparing configuration files in $dir/conf"
  if [ -f $dir/conf/online_decoding.conf ]; then
    echo "$0: moving $dir/conf/online_decoding.conf to $dir/conf/online_decoding.conf.bak"
    mv $dir/conf/online_decoding.conf $dir/conf/online_decoding.conf.bak
  fi
  conf=$dir/conf/online_decoding.conf
  echo -n >$conf
  case "$feature_type" in
    mfcc)
      echo "$0: creating $dir/conf/mfcc.conf"
      echo "--mfcc-config=$dir/conf/mfcc.conf" >>$conf
      cp conf/mfcc.conf $dir/conf/ ;;
    plp)
      echo "$0: enabling plp features"
      echo "--feature-type=plp" >>$conf
      echo "$0: creating $dir/conf/plp.conf"
      echo "--plp-config=$dir/conf/plp.conf" >>$conf
      cp conf/plp.conf $dir/conf/ ;;
    *)
      echo "Unknown feature type $feature_type"
  esac
  if ! cp $online_cmvn_config $dir/conf/online_cmvn.conf; then
    echo "$0: error copying online cmvn config to $dir/conf/"
    exit 1;
  fi
  echo "--cmvn-config=$dir/conf/online_cmvn.conf" >>$conf
  if [ -f $dir/final.mat ]; then
    echo "$0: enabling feature splicing"
    echo "--splice-feats" >>$conf
    echo "$0: creating $dir/conf/splice.conf"
    for x in $(cat $dir/splice_opts); do echo $x; done > $dir/conf/splice.conf
    echo "--splice-config=$dir/conf/splice.conf" >>$conf
    echo "$0: enabling LDA"
    echo "--lda-matrix=$dir/final.mat" >>$conf
  else
    echo "$0: enabling deltas"
    echo "--add-deltas" >>$conf
  fi
  if $add_pitch; then
    echo "$0: enabling pitch features"
    echo "--add-pitch" >>$conf
    echo "$0: creating $dir/conf/pitch.conf"
    echo "--pitch-config=$dir/conf/pitch.conf" >>$conf
    if ! cp $pitch_config $dir/conf/pitch.conf; then
      echo "$0: error copying pitch config to $dir/conf/"
      exit 1;
    fi;
    echo "$0: creating $dir/conf/pitch_process.conf"
    echo "--pitch-process-config=$dir/conf/pitch_process.conf" >>$conf
    if ! cp $pitch_process_config $dir/conf/pitch_process.conf; then
      echo "$0: error copying pitch process config to $dir/conf/"
      exit 1;
    fi;
    nfields=$(sed -n '2,2p' $dir/global_cmvn.stats | \
      perl -e '$_ = <>; s/^\s+|\s+$//g; print scalar(split);');
    if [ $nfields != 17 ]; then
      echo "$0: $dir/global_cmvn.stats has $nfields entries per row (expected 17)."
      echo "$0: Did you append pitch features?"
      exit 1;
    fi
    #offset=$(sed -n '2,2p' $dir/global_cmvn.stats | \
    #  perl -e '$_ = <>; s/^\s+|\s+$//g; ($t, $c) = (split)[13, 16]; print -$t/$c;');
    #echo "--pov-offset=$offset" >>$dir/conf/pitch_process.conf
  fi

  echo "--fmllr-basis=$dir/fmllr.basis" >>$conf
  echo "--online-alignment-model=$dir/final.oalimdl" >>$conf
  echo "--model=$dir/final.mdl" >>$conf
  if ! cmp --quiet $dir/final.mdl $dir/final.rescore_mdl; then
    echo "--rescore-model=$dir/final.rescore_mdl" >>$conf
  fi
  echo "--silence-phones=$silphonelist" >>$conf
  echo "--endpoint.silence-phones=$silphonelist" >>$conf
  echo "--global-cmvn-stats=$dir/global_cmvn.stats" >>$conf
  echo "$0: created config file $conf"
fi
