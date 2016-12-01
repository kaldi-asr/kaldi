#! /bin/bash

# Copyright 2016  Xiaohui Zhang
#           2016  Vimal Manohar
# Apache 2.0

# This script demonstrate how to expand a existing lexicon using a combination
# of acoustic evidence and G2P to learn a lexicon that covers words in a target 
# vocab, and agrees sufficiently with the acoustics. The basic idea is to 
# run phonetic decoding on acoustic training data using an existing
# acoustice model (possibly re-trained using a G2P-expanded lexicon) to get 
# alternative pronunciations for words in training data. Then we combine three
# exclusive sources of pronunciations: the reference lexicon (supposedly 
# hand-derived), phonetic decoding, and G2P (optional) into one lexicon and then run 
# lattice alignment on the same data, to collect acoustic evidence (soft
# counts) of all pronunciations. Based on these statistics, and
# user-specified prior-counts (parameterized by prior mean and prior-counts-tot,
# assuming the prior follows a Dirichlet distribution), we then use a Bayesian
# framework to compute posteriors of all pronunciations for each word,
# and then select best pronunciations for each word. The output is a final learned lexicon
# whose vocab matches the user-specified target-vocab, and two intermediate resultis:
# an edits file which records the recommended changes to all in-ref-vocab words'
# prons, and a half-learned lexicon where all in-ref-vocab words' prons were untouched
# (on top of which we apply the edits file to produce the final learned lexicon).
# The user can always modify the edits file manually and then re-apply it on the
# half-learned lexicon using steps/dict/apply_lexicon_edits to produce the final
# learned lexicon. See the last stage in this script for details.


stage=0
# Begin configuration section.  
cmd=run.pl
nj=
stage=6
oov_symbol=
lexicon_g2p=
min_prob=0.3
variants_prob_mass=0.7
variants_prob_mass_ref=0.9
prior_counts_tot=15
prior_mean="0.7,0.2,0.1"
num_gauss=
num_leaves=
retrain_src_mdl=true
cleanup=true
# End configuration section.  

. ./path.sh
. utils/parse_options.sh

if [ $# -ne 7 ]; then
  echo "Usage: $0 [options] <ref-dict> <target-vocab> <data> <src-mdl-dir> \\"
  echo "          <ref-lang> <dest-dict>."
  echo "  This script does lexicon expansion using a combination of acoustic"
  echo "  evidence and G2P to produce a lexicon that covers words of a target vocab:"
  echo ""               
  echo "Arguments:"
  echo " <ref-dict>     the dir which contains the reference lexicon (most probably hand-derived)"
  echo "                we want to expand/improve, and nonsilence_phones.txt,.etc which we need " 
  echo "                for building new dict dirs."
  echo " <target-vocab> the vocabulary we want the final learned lexicon to cover (one word per line)."
  echo " <data>         acoustic training data we use to get alternative"
  echo "                pronunciations and collet acoustic evidence."
  echo " <src-mdl-dir>  The dir containing an SAT-GMM acoustic model (we optionaly we re-train it" 
  echo "                using G2P expanded lexicon) to do phonetic decoding (to get alternative"
  echo "                pronunciations) and lattice-alignment (to collect acoustic evidence for"
  echo "                evaluating all prounciations)"
  echo " <ref-lang>     the reference lang dir which we use to get non-scored-words"
  echo "                like <UNK> for building new dict dirs"
  echo " <dest-dict>    the dict dir where we put the final learned lexicon, whose vocab"
  echo "                matches <target-vocab>."
  echo ""
  echo "Note: <target-vocab> and the vocab of <data> don't have to match. For words"
  echo "     who are in <target-vocab> but not seen in <data>, their pronunciations" 
  echo "     will be given by G2P at the end."
  echo ""
  echo "e.g. $0 data/local/dict data/local/lm/librispeech-vocab.txt data/train \\"
  echo "          exp/tri3 data/lang data/local/dict_learned"
  echo "Options:"
  echo "  --stage <n>                  # stage to run from, to enable resuming from partially"
  echo "                               # completed run (default: 0)"
  echo "  --cmd '$cmd'                 # command to submit jobs with (e.g. run.pl, queue.pl)"
  echo "  --nj <nj>                    # number of parallel jobs"
  echo "  --oov-symbol '$oov_symbol'   # oov symbol, like <UNK>."
  echo "  --g2p-pron-candidates        # A lexicon file containing g2p generated pronunciations, for words in acoustic training "
  echo "                               # data / target vocabulary. It's optional."
  echo "  --min-prob <float>           # The cut-off parameter used to select pronunciation candidates from phonetic"
  echo "                               # decoding. We remove pronunciations with probabilities less than this value"
  echo "                               # after normalizing the probs s.t. the max-prob is 1.0 for each word."
  echo "  --prior-mean                 # Mean of priors (summing up to 1) assigned to three exclusive pronunciation"
  echo "         <float,float,float>   # source: reference lexicon, g2p, and phonetic decoding (used in the Bayesian"
  echo "                               # pronunciation selection procedure). We recommend setting a larger prior"
  echo "                               # mean for the reference lexicon, e.g. '0.6,0.2,0.2'."
  echo "  --prior-counts-tot <float>   # Total amount of prior counts we add to all pronunciation candidates of"
  echo "                               # each word. By timing it with the prior mean of a source, and then dividing"
  echo "                               # by the number of candidates (for a word) from this source, we get the"
  echo "                               # prior counts we actually add to each candidate."
  echo "  --variants-prob-mass <float> # In the Bayesian pronunciation selection procedure, for each word, we"
  echo "                               # choose candidates (from all three sources) with highest posteriors"
  echo "                               # until the total prob mass hit this amount."
  echo "                               # It's used in a similar fashion when we apply G2P."
  echo "  --variants-prob-mass-ref     # In the Bayesian pronunciation selection procedure, for each word,"
  echo "                               # after the total prob mass of selected candidates hit variants-prob-mass,"
  echo "                               # we continue to pick up reference candidates with highest posteriors"
  echo "                               # until the total prob mass hit this amount (must >= variants-prob-mass)."
  echo "  --num-gauss                  # number of gaussians for the re-trained SAT model (on top of <src-mdl-dir>)."            
  echo "  --num-leaves                 # number of leaves for the re-trained SAT model (on top of <src-mdl-dir>)." 
  echo "  --retrain-src-mdl            # true if you want to re-train the src_mdl before phone decoding (default false)."
  exit 1
fi

echo "$0 $@"  # Print the command line for logging

ref_dict=$1
target_vocab=$2
data=$3
src_mdl_dir=$4
ref_lang=$5
dest_dict=$6
dir=$7 # Most intermediate outputs will be put here. 

mkdir -p $dir
if [ $stage -le 0 ]; then
  echo "$0: Some preparatory work."
  # Get the word counts of training data.
  awk '{for (n=2;n<=NF;n++) counts[$n]++;} END{for (w in counts) printf "%s %d\n",w, counts[w];}' \
    $data/text | sort > $dir/train_counts.txt
  
  # Get the non-scored entries and exclude them from the reference lexicon/vocab, and target_vocab.
  steps/cleanup/internal/get_non_scored_words.py $ref_lang > $dir/non_scored_words
  awk 'NR==FNR{a[$1] = 1; next} {if($1 in a) print $0}' $dir/non_scored_words \
    $ref_dict/lexicon.txt > $dir/non_scored_entries 

  # Remove non-scored-words from the reference lexicon.
  awk 'NR==FNR{a[$1] = 1; next} {if(!($1 in a)) print $0}' $dir/non_scored_words \
    $ref_dict/lexicon.txt | tr -s '\t' ' ' > $dir/ref_lexicon.txt

  cat $dir/ref_lexicon.txt | awk '{print $1}' | sort | uniq > $dir/ref_vocab.txt
  awk 'NR==FNR{a[$1] = 1; next} {if(!($1 in a)) print $0}' $dir/non_scored_words \
    $target_vocab | sort | uniq > $dir/target_vocab.txt
    
  # From the reference lexicon, we estimate the target_num_prons_per_word as,
  # ceiling(avg. # prons per word in the reference lexicon). This'll be used as 
  # the upper bound of # pron variants per word when we apply G2P or select prons to
  # construct the learned lexicon in later stages.
  python -c 'import sys; import math; print int(math.ceil(float(sys.argv[1])/float(sys.argv[2])))' \
    `wc -l $dir/ref_lexicon.txt | awk '{print $1}'` `wc -l $dir/ref_vocab.txt | awk '{print $1}'` \
    > $dir/target_num_prons_per_word || exit 1;

  if [ -z $lexicon_g2p ]; then
    # create an empty list of g2p generated prons, if it's not given.
    touch $dir/lexicon_g2p.txt
  else
    cp $lexicon_g2p $dir/lexicon_g2p.txt 2>/dev/null
  fi
fi

if [ $stage -le 1 ] && $retrain_src_mdl; then
  echo "$0: Expand the reference lexicon to cover all words in the target vocab. and then"
  echo "   ... re-train the source acoustic model for phonetic decoding. "
  mkdir -p $dir/dict_expanded_target_vocab
  cp $ref_dict/{extra_questions.txt,optional_silence.txt,nonsilence_phones.txt,silence_phones.txt} \
    $dir/dict_expanded_target_vocab  2>/dev/null
  rm $dir/dict_expanded_target_vocab/lexiconp.txt $dir/dict_expanded_target_vocab/lexicon.txt 2>/dev/null
  
  # Get the oov words list (w.r.t ref vocab) which are in the target vocab. 
  awk 'NR==FNR{a[$1] = 1; next} !($1 in a)' $dir/ref_lexicon.txt \
    $dir/target_vocab.txt | sort | uniq > $dir/oov_target_vocab.txt

  # Assign pronunciations from lexicon_g2p.txt to oov_target_vocab. For words which
  # cannot be found in lexicon_g2p.txt, we simply ignore them.
  awk 'NR==FNR{a[$1] = 1; next} ($1 in a)' $dir/oov_target_vocab.txt \
    $dir/lexicon_g2p.txt > $dir/lexicon_g2p_oov_target_vocab.txt
  
  cat $dir/lexicon_g2p_oov_target_vocab.txt $dir/ref_lexicon.txt | \
    awk 'NR==FNR{a[$1] = 1; next} ($1 in a)' $dir/target_vocab.txt - | \
    cat $dir/non_scored_entries - | 
    sort | uniq > $dir/dict_expanded_target_vocab/lexicon.txt
  
  utils/prepare_lang.sh --phone-symbol-table $ref_lang/phones.txt $dir/dict_expanded_target_vocab \
    $oov_symbol $dir/lang_expanded_target_vocab_tmp $dir/lang_expanded_target_vocab || exit 1;
  
  # Align the acoustic training data using the given src_mdl_dir.
  alidir=${src_mdl_dir}_ali_$(basename $data) 
  steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" \
    $data $dir/lang_expanded_target_vocab $src_mdl_dir $alidir || exit 1;
  
  # Train another SAT system on the given data and put it in $dir/${src_mdl_dir}_retrained
  # this model will be used for phonetic decoding and lattice alignment later on.
  if [ -z $num_leaves ] || [ -z $num_gauss ] ; then
    echo "num_leaves and num_gauss need to be specified." && exit 1;
  fi
  steps/train_sat.sh --cmd "$train_cmd" $num_leaves $num_gauss \
    $data $dir/lang_expanded_target_vocab $alidir $dir/${src_mdl_dir}_retrained || exit 1;
fi

if [ $stage -le 2 ]; then
  echo "$0: Expand the reference lexicon to cover all words seen in,"
  echo "  ... acoustic training data, and prepare corresponding dict and lang directories."
  echo "  ... This is needed when generate pron candidates from phonetic decoding."
  mkdir -p $dir/dict_expanded_train
  cp $ref_dict/{extra_questions.txt,optional_silence.txt,nonsilence_phones.txt,silence_phones.txt} \
    $dir/dict_expanded_train 2>/dev/null
  rm $dir/dict_expanded_train/lexiconp.txt $dir/dict_expanded_train/lexicon.txt 2>/dev/null

  # Get the oov words list (w.r.t ref vocab) which are in training data. 
  awk 'NR==FNR{a[$1] = 1; next} {if(!($1 in a)) print $1}' $dir/ref_lexicon.txt \
    $dir/train_counts.txt | sort > $dir/oov_train.txt 
  
  awk 'NR==FNR{a[$1] = 1; next} {if(($1 in a)) b+=$2; else c+=$2} END{print c/(b+c)}' \
    $dir/ref_vocab.txt $dir/train_counts.txt > $dir/train_oov_rate
  
  echo "OOV rate (w.r.t. the reference lexicon) of the acoustic training data is:"
  cat $dir/train_oov_rate

  # Assign pronunciations from lexicon_g2p to oov_train. For words which
  # cannot be found in lexicon_g2p, we simply assign oov_symbol's pronunciaiton
  # (like NSN) to them, in order to get phonetic decoding pron candidates for them later on.
  awk 'NR==FNR{a[$1] = 1; next} ($1 in a)' $dir/oov_train.txt \
    $dir/lexicon_g2p.txt > $dir/g2p_prons_for_oov_train.txt
  
  # Get the pronunciation of oov_symbol.
  oov_pron=`cat $dir/non_scored_entries | grep $oov_symbol | cut -f2- -d' '`
  # For oov words in training data for which we don't even have G2P pron candidates,
  # we simply assign them the pronunciation of the oov symbol (like <unk>).
  awk 'NR==FNR{a[$1] = 1; next} {if(!($1 in a)) print $1}' $dir/g2p_prons_for_oov_train.txt \
    $dir/oov_train.txt | awk -v op=$oov_pron '{print $0" "op}' > $dir/oov_train_no_pron.txt
    
  cat $dir/oov_train_no_pron.txt $dir/g2p_prons_for_oov_train.txt $dir/ref_lexicon.txt | \
    awk 'NR==FNR{a[$1] = 1; next} ($1 in a)' $dir/train_counts.txt - | \
    cat - $dir/non_scored_entries | \
    sort | uniq > $dir/dict_expanded_train/lexicon.txt || exit 1;
  
  utils/prepare_lang.sh $dir/dict_expanded_train $oov_symbol \
    $dir/lang_expanded_train_tmp $dir/lang_expanded_train || exit 1;
fi

if [ $stage -le 3 ]; then
  echo "$0: Generate pronunciation candidates from phonetic decoding on acoustic training data.."
  if $retrain_src_mdl; then mdl_dir=$dir/${src_mdl_dir}_retrained; else mdl_dir=$src_mdl_dir; fi
  steps/cleanup/debug_lexicon.sh --nj $nj --cmd "$decode_cmd" $data $dir/lang_expanded_train \
    $mdl_dir $dir/dict_expanded_train/lexicon.txt $dir/phonetic_decoding || exit 1;
  
  # We prune the phonetic decoding generated prons relative to the largest count, by setting "min_prob",
  # and only leave prons who are not present in the reference lexicon / g2p-generated lexicon.
  cat $dir/ref_lexicon.txt $dir/lexicon_g2p.txt > $dir/phonetic_decoding/filter_lexicon.txt 
  
  $cmd $dir/phonetic_decoding/log/prons_to_lexicon.log steps/dict/prons_to_lexicon.py \
    --min-prob=$min_prob --filter-lexicon=$dir/phonetic_decoding/filter_lexicon.txt \
    $dir/phonetic_decoding/prons.txt $dir/lexicon_phonetic_decoding_with_eps.txt
  cat $dir/lexicon_phonetic_decoding_with_eps.txt | grep -vP "<eps>|<UNK>|<unk>|\[.*\]" | \
    sort | uniq > $dir/lexicon_phonetic_decoding.txt || exit 1;
fi

if [ $stage -le 4 ]; then
  echo "$0: Combine the reference lexicon and pronunciations from phone-decoding/G2P into one"
  echo "  ... lexicon, and run lattice alignment using this lexicon on acoustic training data"
  echo "  ... to collect acoustic evidence."
  # Combine the reference lexicon, pronunciations from G2P and phonetic decoding into one lexicon.
  mkdir -p $dir/dict_combined_iter1
  cp $ref_dict/{extra_questions.txt,optional_silence.txt,nonsilence_phones.txt,silence_phones.txt} \
    $dir/dict_combined_iter1/ 2>/dev/null
  rm $dir/dict_combined_iter1/lexiconp.txt $dir/dict_combined_iter1/lexicon.txt 2>/dev/null

  # Filter out words which don't appear in the acoustic training data
  cat $dir/lexicon_phonetic_decoding.txt $dir/lexicon_g2p.txt \
    $dir/ref_lexicon.txt | tr -s '\t' ' ' | \
    awk 'NR==FNR{a[$1] = 1; next} ($1 in a)' $dir/train_counts.txt - | \
    cat $dir/non_scored_entries - | \
    sort | uniq > $dir/dict_combined_iter1/lexicon.txt
  
  utils/prepare_lang.sh --phone-symbol-table $ref_lang/phones.txt \
    $dir/dict_combined_iter1 $oov_symbol \
    $dir/lang_combined_iter1_tmp $dir/lang_combined_iter1 || exit 1;
  
  # Generate lattices for the acoustic training data with the combined lexicon.
  if $retrain_src_mdl; then mdl_dir=$dir/${src_mdl_dir}_retrained; else mdl_dir=$src_mdl_dir; fi
  steps/align_fmllr_lats.sh --cmd "$decode_cmd" --nj $nj \
    $data $dir/lang_combined_iter1 $mdl_dir $dir/lats_iter1 || exit 1;

  # Get arc level information from the lattice.
  $cmd JOB=1:$nj $dir/lats_iter1/log/get_arc_info.JOB.log \
    lattice-align-words $dir/lang_combined_iter1/phones/word_boundary.int \
    $dir/lats_iter1/final.mdl \
    "ark:gunzip -c $dir/lats_iter1/lat.JOB.gz |" ark:- \| \
    lattice-arc-post --acoustic-scale=0.1 $dir/lats_iter1/final.mdl ark:- - \| \
    utils/int2sym.pl -f 5 $dir/lang_combined_iter1/words.txt \| \
    utils/int2sym.pl -f 6- $dir/lang_combined_iter1/phones.txt '>' \
    $dir/lats_iter1/arc_info_sym.JOB.txt || exit 1;
  
  # Get soft counts of all pronunciations from arc level information.
  cat $dir/lats_iter1/arc_info_sym.*.txt | steps/dict/get_pron_stats.py - \
    $dir/phonetic_decoding/phone_map.txt $dir/lats_iter1/pron_stats.txt || exit 1;
fi

if [ $stage -le 5 ]; then
  echo "$0: Prune the pronunciation candidates generated from G2P/phonetic decoding, and re-do lattice-alignment."
  mkdir -p $dir/dict_combined_iter2
  cp $ref_dict/{extra_questions.txt,optional_silence.txt,nonsilence_phones.txt,silence_phones.txt} \
    $dir/dict_combined_iter2/ 2>/dev/null
  rm $dir/dict_combined_iter2/lexiconp.txt $dir/dict_combined_iter2/lexicon.txt 2>/dev/null

  # Prune away pronunciations which have low acoustic evidence from the first pass of lattice alignment.
  $cmd $dir/lats_iter1/log/prune_pron_candidates.log steps/dict/internal/prune_pron_candidates.py $dir/lats_iter1/pron_stats.txt $dir/ref_lexicon.txt $dir/pruned_prons.txt
 
  awk 'NR==FNR{a[$0] = 1; next} (!($0 in a))' $dir/pruned_prons.txt $dir/lexicon_phonetic_decoding.txt \
    > $dir/lexicon_phonetic_decoding_pruned.txt

  awk 'NR==FNR{a[$0] = 1; next} (!($0 in a))' $dir/pruned_prons.txt $dir/lexicon_g2p.txt \
    > $dir/lexicon_g2p_pruned.txt \

  # Filter out words which don't appear in the acoustic training data
  cat $dir/lexicon_phonetic_decoding_pruned.txt $dir/lexicon_g2p_pruned.txt \
    $dir/ref_lexicon.txt | tr -s '\t' ' ' | \
    awk 'NR==FNR{a[$1] = 1; next} ($1 in a)' $dir/train_counts.txt - | \
    cat $dir/non_scored_entries - | \
    sort | uniq > $dir/dict_combined_iter2/lexicon.txt

  utils/prepare_lang.sh --phone-symbol-table $ref_lang/phones.txt \
    $dir/dict_combined_iter2 $oov_symbol \
    $dir/lang_combined_iter2_tmp $dir/lang_combined_iter2 || exit 1;
  
  if $retrain_src_mdl; then mdl_dir=$dir/${src_mdl_dir}_retrained; else mdl_dir=$src_mdl_dir; fi
  steps/align_fmllr_lats.sh --cmd "$decode_cmd" --nj $nj \
    $data $dir/lang_combined_iter2 $mdl_dir $dir/lats_iter2 || exit 1;

  # Get arc level information from the lattice.
  $cmd JOB=1:$nj $dir/lats_iter2/log/get_arc_info.JOB.log \
    lattice-align-words $dir/lang_combined_iter2/phones/word_boundary.int \
    $dir/lats_iter2/final.mdl \
    "ark:gunzip -c $dir/lats_iter2/lat.JOB.gz |" ark:- \| \
    lattice-arc-post --acoustic-scale=0.1 $dir/lats_iter2/final.mdl ark:- - \| \
    utils/int2sym.pl -f 5 $dir/lang_combined_iter2/words.txt \| \
    utils/int2sym.pl -f 6- $dir/lang_combined_iter2/phones.txt '>' \
    $dir/lats_iter2/arc_info_sym.JOB.txt || exit 1;
  
  # Get soft counts of all pronunciations from arc level information.
  cat $dir/lats_iter2/arc_info_sym.*.txt | steps/dict/get_pron_stats.py - \
    $dir/phonetic_decoding/phone_map.txt $dir/lats_iter2/pron_stats.txt || exit 1;
fi

if [ $stage -le 6 ]; then
  echo "$0: Select pronunciations according to the acoustic evidence from lattice alignment."
  # Given the acoustic evidence (soft-counts), we use a Bayesian framework to select pronunciations 
  # from three exclusive candidate sources: reference (hand-derived) lexicon, G2P and phonetic decoding.
  # The posteriors for all candidate prons for all words are printed into pron_posteriors.txt
  # For words which are out of the ref. vocab, the learned prons are written into out_of_ref_vocab_prons_learned.txt.
  # Among them, for words without acoustic evidence, we just ignore them, even if pron candidates from G2P were provided).
  # For words in the ref. vocab, we instead output a human readable & editable "edits" file called
  # ref_lexicon_edits.txt, which records all proposed changes to the prons (if any). Also, a 
  # summary is printed into the log file.
  
  variants_counts=`cat $dir/target_num_prons_per_word` || exit 1;
  $cmd $dir/lats_iter2/log/select_prons_bayesian.log \
    steps/dict/select_prons_bayesian.py --prior-mean=$prior_mean --prior-counts-tot=$prior_counts_tot \
    --variants-counts=$variants_counts --variants-prob-mass=$variants_prob_mass --variants-prob-mass-ref=$variants_prob_mass_ref \
    $ref_dict/silence_phones.txt $dir/lats_iter2/pron_stats.txt $dir/train_counts.txt $dir/ref_lexicon.txt \
    $dir/lexicon_g2p_pruned.txt $dir/lexicon_phonetic_decoding_pruned.txt \
    $dir/lats_iter2/pron_posteriors.temp $dir/lats_iter2/out_of_ref_vocab_prons_learned.txt $dir/lats_iter2/ref_lexicon_edits.txt

  # We reformat the pron_posterior file and add some comments.
  paste <(cat $dir/lats_iter2/pron_posteriors.temp | cut -d' ' -f1-3 | column -t) \
    <(cat $dir/lats_iter2/pron_posteriors.temp | cut -d' ' -f4-) | sort -nr -k1,3 | \
    cat <( echo ';; <word> <source: R(eference)/G(2P)/P(hone-decoding)> <posterior> <pronunciation>') -  \
    > $dir/lats_iter2/pron_posteriors.txt
  rm $dir/pron_posteriors.temp 2>/dev/null

  # Remove some stuff that takes up space and is unlikely to be useful later on.
  if $cleanup; then
    rm -r $dir/lats_iter*/{fsts*,lat*} 2>/dev/null
  fi
fi

if [ $stage -le 7 ]; then
  echo "$0: Expand the learned lexicon further to cover words in target vocab that are."
  echo "  ... not seen in acoustic training data."
  mkdir -p $dest_dict
  cp $ref_dict/{extra_questions.txt,optional_silence.txt,nonsilence_phones.txt,silence_phones.txt} \
    $dest_dict  2>/dev/null
  rm $dest_dict/lexiconp.txt $dest_dict/lexicon.txt 2>/dev/null
  # Get the list of oov (w.r.t. ref vocab) without acoustic evidence, which are in the
  # target vocab. We'll just assign to them pronunciations from lexicon_g2p, if any.
  cat $dir/lats_iter2/out_of_ref_vocab_prons_learned.txt $dir/ref_lexicon.txt | \
    awk 'NR==FNR{a[$1] = 1; next} !($1 in a)' - \
    $dir/target_vocab.txt | sort | uniq > $dir/oov_no_acoustics.txt

  awk 'NR==FNR{a[$1] = 1; next} ($1 in a)' $dir/oov_no_acoustics.txt \
    $dir/lexicon_g2p.txt > $dir/g2p_prons_for_oov_no_acoustics.txt
 
  # We concatenate three lexicons togethers: G2P lexicon for oov words without acoustics,
  # learned lexicon for oov words with acoustics, and the original reference lexicon (for
  # this part, later one we'll apply recommended changes using steps/dict/apply_lexicon_edits.py
  cat $dir/g2p_prons_for_oov_no_acoustics.txt $dir/lats_iter2/out_of_ref_vocab_prons_learned.txt \
    $dir/ref_lexicon.txt | tr -s '\t' ' ' | sort | uniq > $dest_dict/lexicon.temp

  awk 'NR==FNR{a[$1] = 1; next} ($1 in a)' $dir/target_vocab.txt \
    $dest_dict/lexicon.temp | sort | uniq > $dest_dict/lexicon.nosil

  cat $dir/non_scored_entries $dest_dict/lexicon.nosil | sort | uniq >$dest_dict/lexicon0.txt
fi

if [ $stage -le 8 ]; then
  echo "$0: Apply the ref_lexicon_edits file to the reference lexicon."
  echo "  ... The user can inspect/modify the edits file and then re-run:"
  echo "  ... steps/dict/apply_lexicon_edits.py $dest_dict/lexicon0.txt $dir/lats_iter2/ref_lexicon_edits.txt  - | \\"
  echo "  ...   sort -u \> $dest_dict/lexicon.txt to re-produce the final learned lexicon."
  cp $dir/lats_iter2/ref_lexicon_edits.txt $dest_dict/lexicon_edits.txt 2>/dev/null
  steps/dict/apply_lexicon_edits.py $dest_dict/lexicon0.txt $dir/lats_iter2/ref_lexicon_edits.txt - | \
    sort | uniq > $dest_dict/lexicon.txt
fi
