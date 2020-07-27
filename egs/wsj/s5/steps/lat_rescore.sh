#!/bin/bash

echo "$0 $@"

. cmd.sh

# Common opts
cmd="$train_cmd"
stage=0
nj=100
num_threads=
# Lattice-transcripts combination opts
iter=final
#scale_opts="--transition-scale=1.0 --self-loop-scale=1.0"
scale_opts="--transition-scale=0 --self-loop-scale=0"

# Decode opts
acwt=1.0
post_decode_acwt=10.0
lattice_beam=8
online_ivector_dir=
extra_left_context=0
extra_right_context=0
extra_left_context_initial=0
extra_right_context_final=0
frames_per_chunk=50

set -e
. path.sh
. utils/parse_options.sh

if [ $# != 4 ]; then
    echo "Usage $0: [Options] <data> <lang> <decode_dir> <dir>"
    echo "Options"
    echo "    --acwt <acwt>"
    echo "    --post-acwt <acwt>"
    echo "    --lattice-beam <beam>"
    echo "    --online-ivector-dir <dir>"
    echo "    --frames_per_chunk <#frames>"
    echo ""
    exit 1
fi

data=$1
lang=$2
decode_dir=$3
dir=$4
srcdir=`dirname $decode_dir`
tree=$srcdir/tree
model=$srcdir/${iter}.mdl

for f in $data/feats.scp $data/text $lang/G.fst $lang/L_disambig.fst $lang/oov.int $lang/words.txt $decode_dir/lat.1.gz $decode_dir/num_jobs
do
    if [ ! -f $f ]; then
        echo "\"$f\" is expected to exist!"
        exit 1
    fi
done

decode_nj=`cat $decode_dir/num_jobs`
oov=`cat $lang/oov.int`

mkdir -p $dir/log $dir/fst
echo $nj > $dir/num_jobs
sdata=$data/split$nj
[ -d $sdata ] && [ $data/feats.scp -ot $sdata ] || split_data.sh $data $nj || exit 1

reh_wspecifier="ark,scp:$dir/fst/REH.JOB.ark,$dir/fst/reh_fst.JOB.scp"
greh_wspecifier="ark:|gzip -c > $dir/fst/GREH.JOB.gz"
greh_rspecifier="ark:gunzip -c $dir/fst/GREH.JOB.gz|"
graph_wspecifier="ark:|gzip -c > $dir/fst/graph.JOB.gz"
graph_rspecifier="ark:gunzip -c $dir/fst/graph.JOB.gz|"
lat_wspecifier="ark:|lattice-scale --acoustic-scale=$post_decode_acwt ark:- ark:- | gzip -c >$dir/lat.JOB.gz"

if [ $stage -le 1 ]; then
    echo "$0: Doing FST composition of RoEoH."

    sddata=$data/split$decode_nj # split used in decoding

    $cmd JOB=1:$decode_nj $dir/log/raw_composition.JOB.log \
    combine-lattice-transcript \
        "ark:utils/sym2int.pl --map-oov $oov -f 2- $lang/words.txt $sddata/JOB/text|" \
        "ark:gunzip -c $decode_dir/lat.JOB.gz|" \
        "$reh_wspecifier"

    for JOB in `seq $decode_nj`
    do
        cat $dir/fst/reh_fst.${JOB}.scp
    done > $dir/reh_fst.scp

    $cmd JOB=1:$nj $dir/log/filter_raw_scp.JOB.log \
    utils/filter_scp.pl \<\(cut -d\' \' -f1 $sdata/JOB/utt2spk\) \
        $dir/reh_fst.scp \> $sdata/JOB/reh_fst.scp
fi

if [ $stage -le 2 ]; then
    echo "$0: Adding LM weights by composing the FST with G.fst"

    fstarcsort --sort_type="olabel" $lang/G.fst $dir/fst/G.fst
    $cmd JOB=1:$nj $dir/log/lm_composition.JOB.log \
    fsttablecompose $dir/fst/G.fst scp:$sdata/JOB/reh_fst.scp "$greh_wspecifier"
fi

if [ $stage -le 3 ]; then
    echo "$0: Compiling training graphs."

    $cmd JOB=1:$nj $dir/log/compile_train_graphs.JOB.log \
    compile-train-graphs-fsts \
        $scale_opts \
        --read-disambig-syms=$lang/phones/disambig.int \
        $tree $model $lang/L_disambig.fst \
        "$greh_rspecifier" "$graph_wspecifier"
fi

if [ $stage -le 4 ]; then
    echo "$0: Decoding with special graphs."

    cmvn_opts=`cat $srcdir/cmvn_opts` || exit 1
    if [ -f $srcdir/online_cmvn ]; then
        feats="ark,s,cs:apply-cmvn-online $cmvn_opts --spk2utt=ark:$sdata/JOB/spk2utt $srcdir/global_cmvn.stats scp:$sdata/JOB/feats.scp ark:- |"
    else
        feats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- |"
    fi

    ivector_opts=
    if [ ! -z "$online_ivector_dir" ]; then
        ivector_period=$(cat $online_ivector_dir/ivector_period) || exit 1;
        ivector_opts="--online-ivectors=scp:$online_ivector_dir/ivector_online.scp --online-ivector-period=$ivector_period"
    fi

    frame_subsampling_opt=
    if [ -f $srcdir/frame_subsampling_factor ]; then
        # e.g. for 'chain' systems
        frame_subsampling_opt="--frame-subsampling-factor=$(cat $srcdir/frame_subsampling_factor)"
    fi

    $cmd JOB=1:$nj $dir/log/decode.JOB.log \
    nnet3-latgen-faster $ivector_opts $frame_subsampling_opt \
        --frames-per-chunk=$frames_per_chunk \
        --extra-left-context=$extra_left_context \
        --extra-right-context=$extra_right_context \
        --extra-left-context-initial=$extra_left_context_initial \
        --extra-right-context-final=$extra_right_context_final \
        --minimize=false --max-active=7000 --min-active=200 \
        --beam=15.0 --lattice-beam=$lattice_beam --acoustic-scale=$acwt \
        --allow-partial=true --word-symbol-table=$lang/words.txt \
        $model "$graph_rspecifier" "$feats" "$lat_wspecifier"
fi

exit 0

