#!/bin/bash

# Creates FSTs needed for large lm decoding
# And run experiments by invoking steps/decode_tri2a_biglm_faster.sh

# To be run from ..
. path.sh

data=`pwd`/data
graphs=`pwd`/exp/tri2a

for lm_suffix in bg tg_pruned; do

  # These are for building HCLG with #0 on output
  fstproject $data/G_${lm_suffix}.fst \
    | fstarcsort --sort_type=olabel \
    > $graphs/Gr_${lm_suffix}.fst
  echo "Created Gr_${lm_suffix}.fst"

  # These are G^-1 with #0 on input, <eps> on output
  cat $data/G_${lm_suffix}.fst  \
    | fstmap --map_type=invert \
    | fstarcsort --sort_type=ilabel \
    > $graphs/Gm_${lm_suffix}.fst
  echo "Created Gm_${lm_suffix}.fst"

done

for lm_suffix in tg_pruned tg; do

  # These are G' with <eps> on both input and output
  fstproject --project_output=true  $data/G_${lm_suffix}.fst \
    | fstarcsort --sort_type=ilabel \
    > $graphs/Gp_${lm_suffix}.fst
  echo "Created Gp_${lm_suffix}.fst"

done

# Functions to run one experiment defined by $lm1, $lm2

createHCLG() {
  scripts/mkgraph.sh $graphs/Gr_${lm1}.fst exp/tri2a/tree exp/tri2a/final.mdl exp/graph_tri2a_${lm1} 
  echo "Created exp/graph_tri2a_${lm1}/HCLG.fst"
}

runexp() {
  exp=exp/decode_tri2a_${lm1}_composed_${lm2}_eval92
  mkdir -p $exp
  scripts/decode.sh $exp exp/graph_tri2a_${lm1}/HCLG.fst steps/decode_tri2a_composed.sh data/eval_nov92.scp $graphs/Gm_${lm1}.fst $graphs/Gp_${lm2}.fst
  echo "$lm1 $lm2 $exp"
  grep WER $exp/wer
}

# Bigram + pruned trigram
lm1=bg
lm2=tg_pruned
createHCLG
runexp

# Bigram + unpruned trigram
lm1=bg
lm2=tg
createHCLG
runexp

# RESULTS
# exp: decode_tri2a_bg_composed_tg_pruned_eval92  %WER 12.92 4.8 RT
# exp: decode_tri2a_bg_composed_tg_eval92         %WER 12.11 4.0 RT
