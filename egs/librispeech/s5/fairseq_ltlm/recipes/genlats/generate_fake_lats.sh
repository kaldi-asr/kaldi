#!/bin/bash
# Copyright 2021 STC-Innovation LTD (Author: Anton Mitrofanov)

echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh # source the path.
[ -f fairseq_ltlm/path.sh ] && . ./fairseq_ltlm/path.sh 


cmd=utils/run.pl
stage=0
nj=48
seed=$(date +%s)

acwt=1  # Just a default value, used for adaptation and beam-pruning..
post_decode_acwt=17  # can be used in 'chain' systems to scale acoustics by 10 so the
# regular scoring script works.
beam=15.0
max_active=7000
min_active=200
lattice_beam=8.0 # Beam we use in lattice generation.
scoring_opts=
skip_scoring=false
minimize=false
temperature=1.0
filter=cat

prepare_for_lt=false # If true apply rmali and determinize-pruned to lattices
prune_lmscale=19 # only affects if prepare_for_lt is true. lmwt for determinize-pruned
prune_beam=4 # only affects if prepare_for_lt is true. beam for determinize-pruned
# End configuration section.

data=
lang=
tree_dir=
fam_dir=
graph=
dir=


. utils/parse_options.sh || exit 1;

. utils/require_argument_all.sh --data --lang --tree_dir --fam_dir --graph --dir

stretch_model=$fam_dir/sali.pkl
fam_model=$fam_dir/fam.ark

kaldi_model=$tree_dir/final.mdl
oov=`cat $lang/oov.int` || exit 1;


sdata=$data/split${nj}_text
if [ ! -d $sdata ] ; then
    echo "$0:Splitting text"
    for j in $(seq 1 $nj); do [ ! -d $sdata/$j ] && mkdir -p $sdata/$j ; done
    utils/split_scp.pl $data/text $(for j in $(seq 1 $nj); do echo -n $sdata/$j/text" " ; done)
fi
#utils/split_data.sh $data $nj
mkdir -p $dir/log
echo $nj > $dir/num_jobs

if [ $stage -le 1 ] ; then
	echo "$0:Stage 1: Get random ali"
	$cmd JOB=1:$nj $dir/log/random_ali.JOB.log \
		compile-train-graphs  --read-disambig-syms=$lang/phones/disambig.int \
			$tree_dir/tree $kaldi_model $lang/L.fst "ark:sym2int.pl --map-oov $oov -f 2- $lang/words.txt $sdata/JOB/text|" ark,t:- \| \
		python $(dirname $0)/fsts2align.py --seed $seed ark:- \| \
		ali-to-pdf $kaldi_model ark:- ark:"| gzip -c > $dir/ali_pdf.JOB.gz"
fi

if [ $stage -le 2 ] ; then
	echo "$0:Stage 2: Get loglikes and decode"
	lats_wspec="| gzip -c > $dir/lat.JOB.gz"
	if $prepare_for_lt ; then
		prune_acwt=$(echo "print(1/$prune_lmscale)" | python)
		lats_wspec="| lattice-rmali ark:- ark:- | lattice-determinize-pruned --acoustic-scale=$prune_acwt --minimize=true --beam=$prune_beam ark:- ark:- $lats_wspec"
	fi
	$cmd JOB=1:$nj $dir/log/decode.JOB.log \
			gunzip -c $dir/ali_pdf.JOB.gz \| \
			python $(dirname $0)/straight_ali.py --seed=$seed \
				--stretch_model_path $stretch_model ark:- ark:- \| \
			latgen-faster-mapped-fake-am \
				--seed=$seed \
			    --temperature=$temperature \
				--max-active=$max_active \
				--min-active=$min_active \
				--beam=$beam \
				--lattice-beam=$lattice_beam \
				--acoustic-scale=$acwt \
				--allow-partial=true \
				--word-symbol-table=$graph/words.txt \
				$kaldi_model \
				$graph/HCLG.fst \
				ark:$fam_model \
				ark:- ark:- \| \
			lattice-scale --acoustic-scale=$post_decode_acwt ark:- ark:"$lats_wspec" 
fi

$skip_scoring && echo "$0: Skip scoring. Exit." && exit 0

if [ $stage -le 3 ] ; then
	echo "$0:Stage 3: Scoring"
	steps/scoring/score_kaldi_wer.sh $scoring_opts $data $graph $dir

	cat $dir/scoring_kaldi/best_wer
fi

if [ $stage -le 4 ] ; then 
	echo "$0:Stage 4: Oracle wer"
	if $prepare_for_lt ; then
		# stage 2 for skip lattice-depth ( not working this rmali)
		s=2
	else
		s=0
	fi
	steps/oracle_wer.sh --stage $s $data $lang $dir
fi
