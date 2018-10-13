#!/bin/bash

# Copyright 2018        Tien-Hong Lo

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.


# Script for system combination using output of the neural networks.
# This calls nnet3-compute, matrix-sum and latgen-faster-mapped to create a system combination.
set -euo pipefail
# begin configuration section.
cmd=run.pl

# Neural Network
stage=0
iter=final
nj=30
output_name="output"
ivector_scale=1.0
apply_exp=false  # Apply exp i.e. write likelihoods instead of log-likelihoods
compress=false    # Specifies whether the output should be compressed before
                  # dumping to disk
use_gpu=false
skip_diagnostics=false
extra_left_context=0
extra_right_context=0
extra_left_context_initial=-1
extra_right_context_final=-1
online_ivector_dir=
frame_subsampling_factor=
frames_per_chunk=150
average=true

# Decode
beam=15.0 # prune the lattices prior to MBR decoding, for speed.
max_active=7000
min_active=200
acwt=0.1  # Just a default value, used for adaptation and beam-pruning..
post_decode_acwt=1.0  # can be used in 'chain' systems to scale acoustics by 10 so the
                      # regular scoring script works.
lattice_beam=8.0 # Beam we use in lattice generation.
num_threads=1 # if >1, will use latgen-faster--map-parallel
min_lmwt=5
max_lmwt=15
parallel_opts="--num-threads 3"
scoring_opts=
minimize=false
skip_scoring=false

word_determinize=false  # If set to true, then output lattice does not retain
                        # alternate paths a sequence of words (with alternate pronunciations).
                        # Setting to true is the default in steps/nnet3/decode.sh.
                        # However, setting this to false
                        # is useful for generation w of semi-supervised training
                        # supervision and frame-level confidences.
write_compact=true   # If set to false, then writes the lattice in non-compact format,
                     # retaining the acoustic scores on each arc. This is
                     # required to be false for LM rescoring undeterminized
                     # lattices (when --word-determinize is false)
#end configuration section.

[ -f ./path.sh ] && . ./path.sh
. parse_options.sh || exit 1;


if [ $# -lt 5 ]; then
  echo "Usage: $0 [options] <data-dir> <graph-dir> <nnet3-dir> <nnet3-dir2> [<nnet3-dir3> ... ] <output-dir>"
  echo "e.g.:   steps/nnet3/decode_score_fusion.sh --nj 8 \\"
  echo "    --online-ivector-dir exp/nnet3/ivectors_test \\"
  echo "    data/test_hires exp/nnet3/tdnn/graph exp/nnet3/tdnn/output exp/nnet3/tdnn1/output .. \\"
  echo "    exp/nnet3/tdnn_comb/decode_test"
  echo "main options (for others, see top of script file)"
  echo "  --config <config-file>                   # config containing options"
  echo "  --nj <nj>                                # number of parallel jobs"
  echo "  --cmd <cmd>                              # Command to run in parallel with"
  echo "  --iter <iter>                            # Iteration of model to decode; default is final."
  exit 1;
fi

echo "$0 $@"

data=$1
graphdir=$2
dir=${@: -1}  # last argument to the script
shift 2;
model_dirs=( $@ )  # read the remaining arguments into an array
unset model_dirs[${#model_dirs[@]}-1]  # 'pop' the last argument which is odir
num_sys=${#model_dirs[@]}  # number of systems to combine

for f in $graphdir/words.txt $graphdir/phones/word_boundary.int ; do
  [ ! -f $f ] && echo "$0: file $f does not exist" && exit 1;
done

[ ! -z "$online_ivector_dir" ] && \
   extra_files="$online_ivector_dir/ivector_online.scp $online_ivector_dir/ivector_period"
   
if [ ! -z "$online_ivector_dir" ]; then
    ivector_period=$(cat $online_ivector_dir/ivector_period) || exit 1;
    ivector_opts="--online-ivectors=scp:$online_ivector_dir/ivector_online.scp --online-ivector-period=$ivector_period"
fi

# assign frame_subsampling_factor automatically if empty
if [ -z $frame_subsampling_factor ]; then
   frame_subsampling_factor=`cat ${model_dirs[0]}/frame_subsampling_factor` || exit 1;
fi

# check if standard chain system or not.
if [ $frame_subsampling_factor -eq 3 ]; then
   if [ $acwt != 1.0 ] || [ $post_decode_acwt != 10.0 ]; then
     echo -e '\n\n'
     echo "$0 WARNING: In standard chain system, acwt = 1.0, post_decode_acwt = 10.0"
     echo "$0 WARNING: Your acwt = $acwt, post_decode_acwt = $post_decode_acwt"
     echo "$0 WARNING: This is OK if you know what you are doing."
     echo -e '\n\n'
   fi
fi

frame_subsampling_opt=
if [ $frame_subsampling_factor -ne 1 ]; then
  # e.g. for 'chain' systems
  frame_subsampling_opt="--frame-subsampling-factor=$frame_subsampling_factor"
fi

# Possibly use multi-threaded decoder
thread_string=
[ $num_threads -gt 1 ] && thread_string="-parallel --num-threads=$num_threads"

mkdir -p $dir/temp

for i in `seq 0 $[num_sys-1]`; do
  srcdir=${model_dirs[$i]}
  
  model=$srcdir/$iter.mdl
  if [ ! -f $srcdir/$iter.mdl ]; then
    echo "$0: Error: no such file $srcdir/$iter.raw. Trying $srcdir/$iter.mdl exit" && exit 1;
  fi
  
  # check that they have the same tree
  show-transitions $graphdir/phones.txt $model > $dir/temp/transition.${i}.txt
  cmp_tree=`diff -q $dir/temp/transition.0.txt $dir/temp/transition.${i}.txt | awk '{print $5}'`
  if [ ! -z $cmp_tree ]; then
    echo "$0 tree must be the same."
    exit 0;
  fi
  
  # check that they have the same frame-subsampling-factor
  if [ $frame_subsampling_factor -ne `cat $srcdir/frame_subsampling_factor` ]; then
    echo "$0 frame_subsampling_factor must be the same.\\"
    echo "Default:$frame_subsampling_factor \\"
    echo "In $srcdir:`cat $srcdir/frame_subsampling_factor`"
    exit 0;
  fi
  
  for f in $data/feats.scp $model $extra_files; do
    [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
  done

  if [ ! -z "$output_name" ] && [ "$output_name" != "output" ]; then
    echo "$0: Using output-name $output_name"
    model="nnet3-copy --edits='remove-output-nodes name=output;rename-node old-name=$output_name new-name=output' $model - |"
  fi

  ## Set up features.
  if [ -f $srcdir/final.mat ]; then
    echo "$0: Error: lda feature type is no longer supported." && exit 1
  fi
  
  sdata=$data/split$nj;
  cmvn_opts=`cat $srcdir/cmvn_opts` || exit 1;
  
  feats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- |"

  if $apply_exp; then
    output_wspecifier="ark:| copy-matrix --apply-exp ark:- ark:-"
  else
    output_wspecifier="ark:| copy-feats --compress=$compress ark:- ark:-"
  fi

  gpu_opt="--use-gpu=no"
  gpu_queue_opt=

  if $use_gpu; then
    gpu_queue_opt="--gpu 1"
    gpu_opt="--use-gpu=yes"
  fi

  echo "$i $model";
  models[$i]="ark,s,cs:nnet3-compute $gpu_opt $ivector_opts $frame_subsampling_opt \
     --frames-per-chunk=$frames_per_chunk \
     --extra-left-context=$extra_left_context \
     --extra-right-context=$extra_right_context \
     --extra-left-context-initial=$extra_left_context_initial \
     --extra-right-context-final=$extra_right_context_final \
     '$model' '$feats' '$output_wspecifier' |"
done

# remove tempdir
rm -rf $dir/temp

# split data to nj
[[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh $data $nj || exit 1;
echo $nj > $dir/num_jobs


# Assume the nnet trained by 
# the same tree and frame subsampling factor.
mkdir -p $dir/log

if [ -f $model ]; then
  echo "$0: $model exists, copy model to $dir/../"
  cp $model $dir/../
fi

if [ -f $srcdir/frame_shift ]; then
  cp $srcdir/frame_shift $dir/../
  echo "$0: $srcdir/frame_shift exists, copy $srcdir/frame_shift to $dir/../"
elif [ -f $srcdir/frame_subsampling_factor ]; then
  cp $srcdir/frame_subsampling_factor $dir/../
  echo "$0: $srcdir/frame_subsampling_factor exists, copy $srcdir/frame_subsampling_factor to $dir/../"
fi

lat_wspecifier="ark:|"
extra_opts=
if ! $write_compact; then
  extra_opts="--determinize-lattice=false"
  lat_wspecifier="ark:| lattice-determinize-phone-pruned --beam=$lattice_beam --acoustic-scale=$acwt --minimize=$minimize --word-determinize=$word_determinize --write-compact=false $model ark:- ark:- |"
fi

if [ "$post_decode_acwt" == 1.0 ]; then
  lat_wspecifier="$lat_wspecifier gzip -c >$dir/lat.JOB.gz"
else
  lat_wspecifier="$lat_wspecifier lattice-scale --acoustic-scale=$post_decode_acwt --write-compact=$write_compact ark:- ark:- | gzip -c >$dir/lat.JOB.gz"
fi


if [ $stage -le 0 ]; then  
  $cmd --num-threads $num_threads JOB=1:$nj $dir/log/decode.JOB.log \
     matrix-sum --average=$average "${models[@]}" ark:- \| \
     latgen-faster-mapped$thread_string --lattice-beam=$lattice_beam --acoustic-scale=$acwt --allow-partial=true \
     --minimize=$minimize --max-active=$max_active --min-active=$min_active --beam=$beam \
     --word-symbol-table=$graphdir/words.txt ${extra_opts} "$model" \
     $graphdir/HCLG.fst ark:- "$lat_wspecifier"
fi

if [ $stage -le 1 ]; then
  if ! $skip_diagnostics ; then
    [ ! -z $iter ] && iter_opt="--iter $iter"
    steps/diagnostic/analyze_lats.sh --cmd "$cmd" $iter_opt $graphdir $dir
  fi
fi

if ! $skip_scoring ; then
  if [ $stage -le 2 ]; then
    [ ! -x local/score.sh ] && \
    echo "Not scoring because local/score.sh does not exist or not executable." && exit 1;
    echo "score best paths"
    [ "$iter" != "final" ] && iter_opt="--iter $iter"
	scoring_opts="--min_lmwt $min_lmwt"
    local/score.sh $scoring_opts --cmd "$cmd" $data $graphdir $dir
    echo "score confidence and timing with sclite"
  fi
fi


exit 0
