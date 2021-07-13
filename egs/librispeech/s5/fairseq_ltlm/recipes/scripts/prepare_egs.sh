#!/usr/bin/env bash
# Copyright 2021 STC-Innovation LTD (Author: Anton Mitrofanov)

set -euo pipefail
. ./path.sh


cmd=utils/run.pl
stage=0

data_dir=
lats_dir=
lang=

max_len=600

out_dir=
text_fname=text_filtered
word_penalty=0
prune=true
prune_lmscale=19
prune_beam=4

unk='<unk>'
skip_scoring=false
scoring_opts=
filter=

g_fst=
g_fst_weight=0
training_type='oracle_path'
all_oracle_targets=false

help_message="$0 Converting lat.*.gz to dump format
Usage: $0 --data_dir <data_dir> --lats_dir <lats_dir> --lang <lang>
Parameters:
data_dir - kaldi data dir
lats_dir - kaldi decode dir
lang - kaldi lang dir

Extra parameters:
stage - stage:0 - prunning, stage:1 - converting to dump, stage:2 - score_kaldi.sh, stage:3 - oracle_wer.sh
out_dir - dir for printed lattices. Default: <lats_dir>/lt_egs_<training_type>
text_fname - kaldi text file in <data_dir>. Default: text_filtered
word_penalty - added word insertion penalty to lattices. Default: 0
prune - if true then lattices will be punned. Default: true
prune_lmscale - lmscale for pruning. If prune=false this param will be ignored. Default: 19
prune_beam - prunning beam. If prune=false this param will be ignored. Default: 4
unk - unk word. Default: <unk>
ref_filter - special filter for reference text. Default: local/wer_ref_filter
g_fst - G.fst for rescoring lattice weights. Default <lang>/G.fst
g_fst_weight - G.fst weight. Default -0
all_oracle_targets - add all oracle path to targets. Default true
training_type - Training type. Can be 'oracle_path' or 'choices_at_fork'. Default 'oracle_path'
"

. ./utils/parse_options.sh
. ./utils/require_argument_all.sh --data_dir --lats_dir --lang

[ -z $out_dir ] && out_dir=$lats_dir/lt_egs_$training_type
[ -z $g_fst ] && g_fst=$lang/G.fst

[ ! -d $out_dir ] && mkdir -p $out_dir



nj=$(cat $lats_dir/num_jobs)
cat $lats_dir/num_jobs > $out_dir/num_jobs

ref_filtering_cmd="cat"
if [ -z "$filter" ]; then
  [ -x local/wer_output_filter ] && ref_filtering_cmd="local/wer_output_filter"
  [ -x local/wer_ref_filter ] && ref_filtering_cmd="local/wer_ref_filter"
else
  ref_filtering_cmd="$filter"
fi


if [ ! -f $data_dir/text_filtered ] ; then
 cat $data_dir/text | $ref_filtering_cmd > $data_dir/text_filtered
fi

prunned_lats=$lats_dir
if $prune ; then
	if [ $stage -le 0 ] ; then
		prune_acwt=$(echo "print(1/$prune_lmscale)" | python)
		echo "$0: Prune lattices with lmwt=$prune_lmscale beam=$prune_beam"
		$cmd JOB=1:$nj $out_dir/log/lat_convert.JOB.log \
			lattice-copy ark:"gunzip -c $lats_dir/lat.JOB.gz |" ark:- \| \
			lattice-rmali ark:- ark:- \| \
			lattice-add-penalty --word-ins-penalty=$word_penalty ark:- ark:- \| \
			lattice-determinize-pruned --acoustic-scale=$prune_acwt --minimize=true --beam=$prune_beam \
			ark:- ark:"|gzip -c > $out_dir/lat.JOB.gz"
		prunned_lats=$out_dir
	fi
else
	echo "$0: Not prunning. $lats_dir must be already pruned."
	skip_scoring=true # already scored
fi

if [ -f $g_fst ] && [ "$g_fst_weight" != "0" ] ; then
	echo "Applying negative rescoring with lm $g_fst, weight $g_fst_weight"
	lattice_reader="gunzip -c $prunned_lats/lat.JOB.gz | lattice-lmrescore --lm-scale=$g_fst_weight ark:- 'fstproject --project_output=true $g_fst |' ark,t:-"
else
	lattice_reader="gunzip -c $prunned_lats/lat.JOB.gz | lattice-copy ark:- ark,t:- "
fi

tokenizer_opts="--tokenizer_fn $lang/words.txt --unk '$unk'"
ds_opts="--training_type=$training_type --data=- --data_type lat_t --ref_text_fname $data_dir/text_filtered --max_len $max_len"
$all_oracle_targets && ds_opts="--all_oracle_targets $ds_opts"

if [ $stage -le 1 ] ; then
	$cmd JOB=1:$nj $out_dir/log/dump.JOB.log \
    	$lattice_reader \| \
	    python fairseq_ltlm/ltlm/pyscripts/lats_t_to_dump.py $tokenizer_opts \
    	        $ds_opts $out_dir/lat.JOB.dump
fi

$skip_scoring && exit 0

echo "Scoring $prunned_lats"

if [ $stage -le 2 ] ; then
	steps/scoring/score_kaldi_wer.sh $scoring_opts --cmd "$cmd"  $data_dir $lang $prunned_lats
fi

if [ $stage -le 3 ] ; then
	# stage 2 for skip lattice-depth ( not working this rmali)
	steps/oracle_wer.sh --stage 2 $data_dir $lang $prunned_lats
fi
