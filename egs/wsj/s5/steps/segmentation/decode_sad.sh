#! /bin/bash

set -e
set -o pipefail

cmd=run.pl
acwt=0.1
beam=8
max_active=1000

. path.sh

. utils/parse_options.sh

if [ $# -ne 3 ]; then
  echo "Usage: $0 <graph-dir> <log_likes_dir> <decode-dir>"
  echo " e.g.: $0 "
  exit 1 
fi

graph_dir=$1
log_likes_dir=$2
dir=$3

nj=`cat $log_likes_dir/num_jobs`
echo $nj > $dir/num_jobs

for f in $dir/trans.mdl $log_likes_dir/log_likes.1.gz $graph_dir/HCLG.fst; do
  if [ ! -f $f ]; then
    echo "$0: Could not find file $f"
  fi
done

decoder_opts+=(--acoustic-scale=$acwt --beam=$beam --max-active=$max_active)

$cmd JOB=1:$nj $dir/log/decode.JOB.log \
  decode-faster-mapped ${decoder_opts[@]} \
  $dir/trans.mdl \
  $graph_dir/HCLG.fst "ark:gunzip -c $log_likes_dir/log_likes.JOB.gz |" \
  ark:/dev/null ark:- \| \
  ali-to-phones --per-frame $dir/trans.mdl ark:- \
  "ark:|gzip -c > $dir/ali.JOB.gz" 
