#!/bin/bash
# Copyright 2019 Alpha Cephei Inc.
# Copyright 2018 Joan Puigcerver
# Copyright 2010-2012 Microsoft Corporation
#           2012-2013 Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0

# This script creates setup for decoding with lookahead online composition. The 
# graph HCLr.fst represents pronunciation dictionary (lexicon), context-dependency,
# and HMM structure in our model. The graph Gr.fst represents the language model.
# Both are combined into single graph HCLrGr for quick testing.
# See
#  http://kaldi-asr.org/doc/graph_recipe_test.html
# (this is compiled from this repository using Doxygen,
# the source for this part is in src/doc/graph_recipe_test.dox)

set -o pipefail

tscale=1.0
loopscale=0.1

remove_oov=false
compose_graph=false

for x in `seq 4`; do
  [ "$1" == "--mono" -o "$1" == "--left-biphone" -o "$1" == "--quinphone" ] && shift && \
    echo "WARNING: the --mono, --left-biphone and --quinphone options are now deprecated and ignored."
  [ "$1" == "--remove-oov" ] && remove_oov=true && shift;
  [ "$1" == "--compose-graph" ] && compose_graph=true && shift;
  [ "$1" == "--transition-scale" ] && tscale=$2 && shift 2;
  [ "$1" == "--self-loop-scale" ] && loopscale=$2 && shift 2;
done

if [ $# != 3 ]; then
   echo "Usage: $0 [options] <lang-dir> <model-dir> <graphdir>"
   echo "e.g.: $0 data/lang_test exp/tri1 exp/tri1/lgraph"
   echo " Options:"
   echo " --remove-oov       #  If true, any paths containing the OOV symbol (obtained from oov.int"
   echo "                    #  in the lang directory) are removed from the G.fst during compilation."
   echo " --transition-scale #  Scaling factor on transition probabilities."
   echo " --self-loop-scale  #  Please see: http://kaldi-asr.org/doc/hmm.html#hmm_scale."
   echo "Note: the --mono, --left-biphone and --quinphone options are now deprecated"
   echo "and will be ignored."
   exit 1;
fi

if [ -f path.sh ]; then . ./path.sh; fi

lang=$1
tree=$2/tree
model=$2/final.mdl
dir=$3

mkdir -p $dir

required="$lang/L_disambig.fst $lang/G.fst $lang/phones.txt $lang/words.txt $lang/phones/silence.csl $lang/phones/disambig.int $arpa $model $tree"
for f in $required; do
  [ ! -f $f ] && echo "$0 : expected $f to exist" && exit 1;
done

if [ -f $dir/HCLG.fst ]; then
  # detect when the result already exists, and avoid overwriting it.
  must_rebuild=false
  for f in $required; do
    [ $f -nt $dir/HCLG.fst ] && must_rebuild=true
  done
  if ! $must_rebuild; then
    echo "$0: $dir/HCLG.fst is up to date."
    exit 0
  fi
fi

N=$(tree-info $tree | grep "context-width" | cut -d' ' -f2) || { echo "Error when getting context-width"; exit 1; }
P=$(tree-info $tree | grep "central-position" | cut -d' ' -f2) || { echo "Error when getting central-position"; exit 1; }

[[ -f $2/frame_subsampling_factor && "$loopscale" == "0.1" ]] && \
  echo "$0: WARNING: chain models need '--self-loop-scale 1.0'";

trap "rm -f $dir/L_disambig_det.fst.$$" EXIT HUP INT PIPE TERM
# Note: [[ ]] is like [ ] but enables certain extra constructs, e.g. || in
# place of -o
if [[ ! -s $dir/L_disambig_det.fst || $dir/L_disambig_det -ot $lang/L_disambig.fst ]]; then
  fstdeterminizestar --use-log=true $lang/L_disambig.fst | fstarcsort --sort_type=ilabel > $dir/L_disambig_det.fst.$$ || exit 1;
  mv $dir/L_disambig_det.fst.$$ $dir/L_disambig_det.fst
fi

cl=$dir/CL_${N}_${P}.fst
cl_tmp=$cl.$$
ilabels=$dir/ilabels_${N}_${P}
ilabels_tmp=$ilabels.$$
trap "rm -f $cl_tmp $ilabels_tmp" EXIT HUP INT PIPE TERM
if [[ ! -s $cl || $cl -ot $dir/L_disambig_det.fst \
    || ! -s $ilabels || $ilabels -ot $dir/L_disambig_det.fst ]]; then
  fstcomposecontext $nonterm_opt --context-size=$N --central-position=$P \
   --read-disambig-syms=$lang/phones/disambig.int \
   --write-disambig-syms=$dir/disambig_ilabels_${N}_${P}.int \
    $ilabels_tmp $dir/L_disambig_det.fst | \
    fstarcsort --sort_type=ilabel > $cl_tmp
  mv $cl_tmp $cl
  mv $ilabels_tmp $ilabels
  fstisstochastic $cl || echo "[info]: CL not stochastic."
fi

trap "rm -f $dir/Ha.fst.$$" EXIT HUP INT PIPE TERM
if [[ ! -s $dir/Ha.fst || $dir/Ha.fst -ot $model  \
    || $dir/Ha.fst -ot $dir/ilabels_${N}_${P} ]]; then
  make-h-transducer $nonterm_opt --disambig-syms-out=$dir/disambig_tid.int \
    --transition-scale=$tscale $dir/ilabels_${N}_${P} $tree $model | \
  fstarcsort --sort_type=olabel \
     > $dir/Ha.fst.$$  || exit 1;
  mv $dir/Ha.fst.$$ $dir/Ha.fst
fi

trap "rm -f $dir/HCLr.fst.$$" EXIT HUP INT PIPE TERM
if [[ ! -s $dir/HCLr.fst || $dir/HCLr.fst -ot $dir/Ha.fst || \
      $dir/HCLr.fst -ot $cl ]]; then
  fstcompose $dir/Ha.fst "$cl" | fstdeterminizestar --use-log=true | \
     fstrmepslocal --use-log=true | \
     fstminimizeencoded | \
     add-self-loops --disambig-syms=$dir/disambig_tid.int --self-loop-scale=$loopscale --reorder=true $model | \
     fstarcsort --sort_type=olabel | \
     fstconvert --fst_type=olabel_lookahead --save_relabel_opairs=${dir}/relabel \
      > $dir/HCLr.fst.$$ || exit 1;
  mv $dir/HCLr.fst.$$ $dir/HCLr.fst
fi

trap "rm -f $dir/Gr.fst.$$" EXIT HUP INT PIPE TERM
if [[ ! -s $dir/Gr.fst || $dir/Gr.fst -ot $lang/G.fst ]]; then
  gr=${lang}/G.fst
  if $remove_oov; then
    [ ! -f $lang/oov.int ] && \
      echo "$0: --remove-oov option: no file $lang/oov.int" && exit 1;
    fstrmsymbols --remove-arcs=true --apply-to-output=true $lang/oov.int $gr | \
      fstrelabel --relabel_ipairs=${dir}/relabel | \
      fstarcsort --sort_type=ilabel | \
      fstconvert --fst_type=const > ${dir}/Gr.fst.$$
  else
    fstrelabel --relabel_ipairs=${dir}/relabel "$gr" | \
      fstarcsort --sort_type=ilabel | \
      fstconvert --fst_type=const > ${dir}/Gr.fst.$$
  fi
  mv $dir/Gr.fst.$$ $dir/Gr.fst
fi

if $compose_graph; then
  trap "rm -f $dir/HCLG.fst.$$" EXIT HUP INT PIPE TERM
  if [[ ! -s $dir/HCLG.fst || $dir/HCLG.fst -ot $dir/HCLr.fst \
        || $dir/HCLG.fst -ot $dir/Gr.fst ]]; then
    fstcompose ${dir}/HCLr.fst ${dir}/Gr.fst | \
    fstrmsymbols $dir/disambig_tid.int  | \
    fstconvert --fst_type=const > $dir/HCLG.fst.$$ || exit 1;
    mv $dir/HCLG.fst.$$ $dir/HCLG.fst
    if [ $tscale == 1.0 -a $loopscale == 1.0 ]; then
      # No point doing this test if transition-scale not 1, as it is bound to fail.
      fstisstochastic $dir/HCLG.fst || echo "[info]: final HCLG is not stochastic."
    fi
  fi

  # note: the empty FST has 66 bytes.  this check is for whether the final FST
  # is the empty file or is the empty FST.
  if ! [ $(head -c 67 $dir/HCLG.fst | wc -c) -eq 67 ]; then
    echo "$0: it looks like the result in $dir/HCLG.fst is empty"
    exit 1
  fi
fi

# keep a copy of the lexicon and a list of silence phones with HCLG...
# this means we can decode without reference to the $lang directory.

cp $lang/words.txt $dir/ || exit 1;
#utils/mkgraph_lookahead_vocab.py $dir/relabel $lang/words.txt > $dir/words.txt || exit 1;
mkdir -p $dir/phones
cp $lang/phones/word_boundary.* $dir/phones/ 2>/dev/null # might be needed for ctm scoring,
cp $lang/phones/align_lexicon.* $dir/phones/ 2>/dev/null # might be needed for ctm scoring,
cp $lang/phones/optional_silence.* $dir/phones/ 2>/dev/null # might be needed for analyzing alignments.
    # but ignore the error if it's not there.


cp $lang/phones/disambig.{txt,int} $dir/phones/ 2> /dev/null
cp $lang/phones/silence.csl $dir/phones/ || exit 1;
cp $lang/phones.txt $dir/ 2> /dev/null # ignore the error if it's not there.

am-info --print-args=false $model | grep pdfs | awk '{print $NF}' > $dir/num_pdfs
