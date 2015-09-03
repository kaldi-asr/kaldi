#!/bin/bash
# Copyright Brno University of Technology (Author: Karel Vesely) 2014.  Apache 2.0.

# This script converts lattices to HTK format compatible with other toolkits.
# We can choose to put words to nodes or arcs, as both is valid in the SLF format.

# begin configuration section.
cmd=run.pl
dirname=lats-in-htk-slf
parallel_opts="-tc 50" # We should limit disk stress
word_to_node=false # Words in arcs or nodes? [default:arcs]
#end configuration section.

echo "$0 $@"

[ -f ./path.sh ] && . ./path.sh
. parse_options.sh || exit 1;

if [ $# -ne 3 ]; then
  echo "Usage: $0 [options] <data-dir> <lang-dir|graph-dir> <decode-dir>"
  echo " Options:"
  echo "    --cmd (run.pl|queue.pl...)      # specify how to run the sub-processes."
  echo "    --word-to-link (true|false)     # put word symbols on links or nodes."
  echo "    --parallel-opts STR             # parallelization options (def.: '-tc 50')."
  echo "e.g.:"
  echo "$0 data/dev data/lang exp/tri4a/decode_dev"
  exit 1;
fi

data=$1
lang=$2 # Note: may be graph directory not lang directory, but has the necessary stuff copied.
dir=$3

model=$(dirname $dir)/final.mdl # assume model one level up from decoding dir.

for f in $lang/words.txt $lang/phones/word_boundary.int $model $dir/lat.1.gz; do
  [ ! -f $f ] && echo "$0: expecting file $f to exist" && exit 1;
done

[ ! -d $dir/$dirname/log ] && mkdir -p $dir/$dirname

echo "$0: Converting lattices into '$dir/$dirname'"

# Words in arcs or nodes? [default:nodes]
word_to_link_arg=
$word_to_node && word_to_node_arg="--word-to-node"

nj=$(cat $dir/num_jobs)

# convert the lattices (individually, gzipped)
$cmd $parallel_opts JOB=1:$nj $dir/$dirname/log/lat_convert.JOB.log \
  mkdir -p $dir/$dirname/JOB/ '&&' \
  lattice-align-words-lexicon --output-error-lats=true --output-if-empty=true $lang/phones/align_lexicon.int $model "ark:gunzip -c $dir/lat.JOB.gz |" ark,t:- \| \
  utils/int2sym.pl -f 3 $lang/words.txt \| \
  utils/convert_slf.pl $word_to_node_arg - $dir/$dirname/JOB/ || exit 1

# make list of lattices
find -L $PWD/$dir/$dirname -name *.lat.gz > $dir/$dirname/lat_htk.scp || exit 1

# check number of lattices:
nseg=$(cat $data/segments | wc -l)
nlat_out=$(cat $dir/$dirname/lat_htk.scp | wc -l)
echo "segments $nseg, saved-lattices $nlat_out"
#
[ $nseg -ne $nlat_out ] && echo "WARNING: missing $((nseg-nlat_out)) lattices for some segments!" \
  && exit 1

echo "success, converted lats to HTK : $PWD/$dir/$dirname/lat_htk.scp"
exit 0

