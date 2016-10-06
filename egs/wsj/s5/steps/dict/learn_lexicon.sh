#! /bin/bash

# Copyright 2016  Xiaohui Zhang
#           2016  Vimal Manohar
# Apache 2.0

# This script demonstrate how to expand a existing lexicon using a combination
# of acoustic evidence and G2P to learn a lexicon that covers words in a target 
# vocab, and agrees sufficiently with the acoustics. The basic idea is to 
# train a G2P model first, using the given reference lexicon, and then apply the
# G2P model on all oov words in acoustic training data, and then run phone 
# level decoding on the same data and the G2P-expanded lexicon using an existing
# acoustice model (possibly re-trained using a G2P-expanded lexicon) to get 
# alternative pronunciations for all words. Then we combine these three
# exclusive sources of pronunciations: the reference lexicon (supposedly 
# hand-derived), lexicons from G2P/phone decoding into one lexicon and then run 
# lattice alignment on the same data, to collect acoustic evidence (soft
# counts) of all pronunciations. Based on these statistics, and
# user-specified prior-counts (parameterized by prior mean and prior-counts-tot,
# assuming the prior follows a Dirichlet distribution), we then use a Bayesian
# framework to compute posteriors of all pronunciations for each word,
# and then select best pronunciations for each word. The output is a learned lexicon
# whose vocab matches the user-specified target-vocab. By setting apply_edits
# as false, the user can choose to keep pronunciations of words from the reference vocab
# unmodified, and output an edits file which records the recommended changes to all 
# in-vocab words' prons. The user can change the edits file manually and then apply
# it to the learned lexicon using steps/dict/apply_lexicon_edits after running this script.

stage=0

# Begin configuration section.  
cmd=run.pl
nj=
stage=6
oov_symbol=
min_prob=0.3
variants_prob_mass=0.7
variants_prob_mass2=0.9
prior_counts_tot=15
prior_mean="0.7,0.2,0.1"
g2p_for_iv=true
affix="lex"
num_gauss=
num_leaves=
retrain_src_mdl=true
apply_edits=true
cleanup=true
# End configuration section.  

. ./path.sh
. utils/parse_options.sh

if [ $# -ne 6 ]; then
  echo "Usage: $0 [options] <ref-dict> <target-vocab> <data> <src-mdl> \\"
  echo "          <ref-lang> <dest-dict>."
  echo "  This script does lexicon expansion using a combination of acoustic"
  echo "  evidence and G2P to produce a lexicon that covers words of a target vocab:"
  echo ""               
  echo "Arguments:"
  echo " <ref-dict>     the dir which contains the reference lexicon (most probably hand-derived)"
  echo "                we want to expand/improve, and nonsilence_phones.txt,.etc which we need " 
  echo "                for building new dict dirs."
  echo " <target-vocab> the vocabulary we want the final learned lexicon to cover."
  echo " <data>         acoustic training data we use to get alternative"
  echo "                pronunciations and collet acoustic evidence."
  echo " <src-mdl>      the acoustic model based on which we re-train an SAT model" 
  echo "                to do phone level decoding (to get alternative pronunciations)"
  echo "                and lattice-alignment (to collect acoustic evidence for"
  echo "                evaluating all prounciations"
  echo " <ref-lang>     the reference lang dir which we use to get non-scored-words"
  echo "                like <UNK> for building new dict dirs"
  echo " <dest-dict>    the dict dir where we put the final learned lexicon, whose vocab"
  echo "                matches <target-vocab>. If apply_edits is set as false, the"
  echo "                script will keep the prons of the words in ref. vocab unchanged"
  echo "                and put the proposed changes in lexicon_edits.txt in this dir."
  echo "                The user should use steps/dict/apply_lexicon_edits.py to apply it"
  echo "                to edits file to lexicon0.txt to produce the final lexicon.txt."
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
  echo "  --min-prob <float>           # the cut-off parameter used to select pronunciation candidates from phone"
  echo "                               # decoding. A smaller min-prob means more candidates will be included."
  echo "  --prior-mean                 # Mean of priors (summing up to 1) assigned to three exclusive pronunciation"
  echo "         <float,float,float>   # source: reference lexicon, g2p, and phone decoding (used in the Bayesian"
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
  echo "  --variants-prob-mass2<float> # In the Bayesian pronunciation selection procedure, for each word,"
  echo "                               # after the total prob mass of selected candidates hit variants-prob-mass,"
  echo "                               # we continue to pick up reference candidates with highest posteriors"
  echo "                               # until the total prob mass hit this amount."
  echo "  --g2p-for-iv <true|false>    # apply G2P for in-vocab words to get more alternative pronunciations."
  echo "  --affix                      # the affix we want to append to the dir to put the retrained model "
  echo "                               # and all intermediate outputs, like 'lex'."
  echo "  --num-gauss                  # number of gaussians for the re-trained SAT model (on top of <src-mdl>)."            
  echo "  --num-leaves                 # number of leaves for the re-trained SAT model (on top of <src-mdl>)." 
  echo "  --retrain-src-mdl            # whether we want to re-train the src_mdl (If the ref_dict is large enough,"
  echo "                               # we may not want to retrain it.)"
  exit 1
fi

echo "$0 $@"  # Print the command line for logging

ref_dict=$1
target_vocab=$2
data=$3
src_mdl=$4
ref_lang=$5
dest_dict=$6
dir=${src_mdl}_${affix}_work # Most outputs will be put here. 

mkdir -p $dir
if [ $stage -le 0 ]; then
  echo "$0: Train G2P model on the reference lexicon, and do some preparatory work."
  steps/dict/train_g2p.sh --cmd "$decode_cmd --mem 4G" $ref_dict/lexicon.txt $dir/g2p || exit 1;
    
  awk '{for (n=2;n<=NF;n++) counts[$n]++;} END{for (w in counts) printf "%s %d\n",w, counts[w];}' \
    $data/text | sort > $dir/train_counts.txt
  
  steps/cleanup/get_non_scored_words.py $ref_lang > $dir/non_scored_words
  
  awk 'NR==FNR{a[$1] = 1; next} {if($1 in a) print $0}' $dir/non_scored_words \
    $ref_dict/lexicon.txt > $dir/non_scored_entries 

  # Exclude non-scored entries from reference lexicon/vocab, and target_vocab.
  awk 'NR==FNR{a[$1] = 1; next} {if(!($1 in a)) print $0}' $dir/non_scored_words \
    $ref_dict/lexicon.txt | tr -s '\t' ' ' > $dir/ref_lexicon.txt

  cat $dir/ref_lexicon.txt | cut -f1 -d' ' | sort | uniq > $dir/ref_vocab.txt
  awk 'NR==FNR{a[$1] = 1; next} {if(!($1 in a)) print $0}' $dir/non_scored_words \
    $target_vocab | sort | uniq > $dir/target_vocab.txt
    
  # From the reference lexicon, we estimate the target_num_prons_per_word as,
  # ceiling(avg. # prons per word in the reference lexicon). This'll be used as 
  # the upper bound of # pron variants per word when we apply G2P or select prons to
  # construct the learned lexicon in later stages.
  python -c 'import sys; import math; print int(math.ceil(float(sys.argv[1])/float(sys.argv[2])))' \
    `wc -l $dir/ref_lexicon.txt | cut -f1 -d' '` `wc -l $dir/ref_vocab.txt | cut -f1 -d' '` \
    > $dir/target_num_prons_per_word || exit 1;
fi

if [ $stage -le 1 ]; then
  echo "$0: Create G2P-expanded lexicon, which expands the original reference lexicon to all words seen in,"
  echo "  ... acoustic training data, and prepare corresponding dict and lang directories."
  awk 'NR==FNR{a[$1] = 1; next} {if(!($1 in a)) print $1}' $dir/ref_vocab.txt \
    $dir/train_counts.txt | sort > $dir/oov_train.txt 
  
  awk 'NR==FNR{a[$1] = 1; next} {if(($1 in a)) b+=$2; else c+=$2} END{print c/(b+c)}' \
    $dir/ref_vocab.txt $dir/train_counts.txt > $dir/train_oov_rate
  
  # when applying G2P for words from acoustic training data, we double the upper bound of
  # num. pron variants per word, and use a larger var-mass (0.9, which is the default value),
  # cause we want many candidate prons from G2P (and also from phone-decoding later) 
  # which will be evaluated by lattice ailgnment.
  variants_counts=`cat $dir/target_num_prons_per_word` || exit 1;
  steps/dict/apply_g2p.sh --var-counts $[2*$variants_counts] $dir/oov_train.txt \
    $dir/g2p $dir/g2p/oov_lex_train
    
  mkdir -p $dir/dict_g2p_expanded_train
  cp $ref_dict/{extra_questions.txt,optional_silence.txt,nonsilence_phones.txt,silence_phones.txt} \
    $dir/dict_g2p_expanded_train 2>/dev/null
  rm $dir/dict_g2p_expanded_train/lexiconp.txt 2>/dev/null

  cat $dir/g2p/oov_lex_train/lexicon.lex |  cut -f 1,3 -d$'\t' | cat - $dir/ref_lexicon.txt | \
    tr -s '\t' ' ' | sort | uniq > $dir/dict_g2p_expanded_train/lexicon.nosil

  cat $dir/non_scored_entries $dir/dict_g2p_expanded_train/lexicon.nosil | \
    sort | uniq > $dir/dict_g2p_expanded_train/lexicon.txt || exit 1;
  
  utils/prepare_lang.sh $dir/dict_g2p_expanded_train $oov_symbol \
    $dir/lang_g2p_train_local $dir/lang_g2p_train || exit 1;
fi

if [ $stage -le 2 ] && $retrain_src_mdl; then
  echo "$0: Make a lexicon that covers all words in the target vocab"
  
  mkdir -p $dir/g2p/oov_target_vocab
  mkdir -p $dir/dict_g2p_target_vocab
  awk 'NR==FNR{a[$1] = 1; next} !($1 in a)' $dir/dict_g2p_expanded_train/lexicon.txt \
    $dir/target_vocab.txt | sort | uniq > $dir/oov_target_vocab.txt
  
  variants_counts=`cat $dir/target_num_prons_per_word` || exit 1;
  steps/dict/apply_g2p.sh --var-counts $variants_counts --var-mass $variants_prob_mass $dir/oov_target_vocab.txt \
    $dir/g2p $dir/g2p/oov_lex_target_vocab || exit 1;
  
  mkdir -p $dir/dict_g2p_target_vocab
  cp $ref_dict/{extra_questions.txt,optional_silence.txt,nonsilence_phones.txt,silence_phones.txt} \
    $dir/dict_g2p_target_vocab  2>/dev/null
  rm $dir/dict_g2p_target_vocab/lexiconp.txt 2>/dev/null

  cat $dir/g2p/oov_lex_target_vocab/lexicon.lex |  cut -f 1,3 -d$'\t' | \
    cat - $dir/dict_g2p_expanded_train/lexicon.txt | tr -s '\t' ' ' | sort | uniq \
    > $dir/dict_g2p_target_vocab/lexicon.temp

  awk 'NR==FNR{a[$1] = 1; next} ($1 in a)' $dir/target_vocab.txt \
    $dir/dict_g2p_target_vocab/lexicon.temp | sort | uniq > $dir/dict_g2p_target_vocab/lexicon.nosil

  cat $dir/non_scored_entries $dir/dict_g2p_target_vocab/lexicon.nosil | sort | \
    uniq > $dir/dict_g2p_target_vocab/lexicon.txt
  
  utils/prepare_lang.sh --phone-symbol-table $ref_lang/phones.txt $dir/dict_g2p_target_vocab
    $oov_symbol $dir/lang_g2p_target_vocab_local $dir/lang_g2p_target_vocab || exit 1;
fi

if [ $stage -le 3 ] && $retrain_src_mdl; then
  echo "$0: Re-train the given acoustic model using the above lexicon and given acoustic training data." 
  # Align the acoustic training data using the given src_mdl.
  alidir=${src_mdl}_ali_$(basename $data) 
  steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" \
    $data $dir/lang_g2p_target_vocab $src_mdl $alidir || exit 1;
  
  # Train another SAT system on the given data and put it in ${src_mdl}_${affix},
  # this model will be used for phone decoding and lattice alignment later on.
  steps/train_sat.sh --cmd "$train_cmd" $num_leaves $num_gauss \
    $data $dir/lang_g2p_target_vocab $alidir ${src_mdl}_${affix} || exit 1;
fi

if [ $stage -le 4 ]; then
  echo "$0: Run phone-level decoding on acoustic training data."
  mdl=$src_mdl && $retrain_src_mdl && mdl=${src_mdl}_${affix}
  steps/cleanup/debug_lexicon.sh --nj $nj --cmd "$decode_cmd" $data $dir/lang_g2p_train \
    $mdl $dir/dict_g2p_expanded_train/lexicon.txt $dir/phone_decode || exit 1;
fi

if [ $stage -le 5 ]; then
  echo "$0: Get alternative pronunciations from phone decoding results (for both oov and iv words), and from G2P for iv words (optional)"
  # Optionally, we apply G2P on in-vocab words to get more alternative pronunciations,
  # since we've found this helpful in some cases.
  if $g2p_for_iv; then
    echo "$0: Applying G2P on in-vocab words as requested."
    variants_counts=`cat $dir/target_num_prons_per_word` || exit 1;
    steps/dict/apply_g2p.sh --var-counts $[2*$variants_counts] $dir/ref_vocab.txt \
      $dir/g2p $dir/g2p/iv_lex || exit 1;
    cat $dir/g2p/iv_lex/lexicon.lex |  cut -f 1,3 -d$'\t' | cat - $dir/dict_g2p_expanded_train/lexicon.nosil | \
      tr -s '\t' ' ' | awk 'NR==FNR{gsub(/\t/," ",$0); a[$0] = 1; next} {if(!($0 in a)) print $0}' \
      $dir/ref_lexicon.txt - | sort | uniq > $dir/g2p_generated_prons.txt || exit 1; 
  else
    cat $dir/dict_g2p_expanded_train/lexicon.nosil | \
      awk 'NR==FNR{gsub(/\t/," ",$0); a[$0] = 1; next} {if(!($0 in a)) print $0}' $dir/ref_lexicon.txt - | \
      sort | uniq > $dir/g2p_generated_prons.txt || exit 1; 
  fi

  # We prune the phone-decoding generated prons relative to the largest count, by setting "min_prob",
  # and only leave prons who are not present in the reference lexicon / g2p-generated lexicon.
  cat $dir/ref_lexicon.txt $dir/g2p_generated_prons.txt > $dir/phone_decode/filter_lexicon.txt 
  $cmd $dir/phone_decode/log/prons_to_lexicon.log steps/dict/prons_to_lexicon.py \
    --min-prob=$min_prob --filter-lexicon=$dir/phone_decode/filter_lexicon.txt \
    $dir/phone_decode/prons.txt $dir/phone_decoding_generated_prons_unfiltered.txt
  cat $dir/phone_decoding_generated_prons_unfiltered.txt | grep -vP "<eps>|<UNK>|<unk>|\[.*\]" | \
    sort | uniq > $dir/phone_decoding_generated_prons.txt || exit 1;
fi

if [ $stage -le 6 ]; then
  echo "$0: Combine the reference lexicon and pronunciations from phone-decoding/G2P into one"
  echo "  ... lexicon, and run lattice alignment using this lexicon on acoustic training data."
  
  mkdir -p $dir/dict_combined
  cp $ref_dict/{extra_questions.txt,optional_silence.txt,nonsilence_phones.txt,silence_phones.txt} \
    $dir/dict_combined/ 2>/dev/null
  rm $dir/dict_combined/lexiconp.txt 2>/dev/null

  cat $dir/phone_decoding_generated_prons.txt $dir/g2p_generated_prons.txt \
    $dir/ref_lexicon.txt | tr -s '\t' ' ' | \
    awk 'NR==FNR{a[$1] = 1; next} ($1 in a)' $dir/train_counts.txt - | \
    cat $dir/non_scored_entries - | \
    sort | uniq > $dir/dict_combined/lexicon.txt
  
  utils/prepare_lang.sh --phone-symbol-table $ref_lang/phones.txt \
    $dir/dict_combined $oov_symbol \
    $dir/lang_combined_local $dir/lang_combined || exit 1;
  
  mdl=$src_mdl && $retrain_src_mdl && mdl=${src_mdl}_${affix}
  steps/align_fmllr_lats.sh --cmd "$decode_cmd" --nj $nj \
    $data $dir/lang_combined $mdl $dir/lats || exit 1;

  # Get arc level information from the lattice.
  $cmd JOB=1:$nj $dir/lats/log/get_arc_info.JOB.log \
    lattice-align-words $dir/lang_combined/phones/word_boundary.int \
    $dir/lats/final.mdl \
    "ark:gunzip -c $dir/lats/lat.JOB.gz |" ark:- \| \
    lattice-arc-post --acoustic-scale=0.1 $dir/lats/final.mdl ark:- - \| \
    utils/int2sym.pl -f 5 $dir/lang_combined/words.txt \| \
    utils/int2sym.pl -f 6- $dir/lang_combined/phones.txt '>' \
    $dir/lats/arc_info_sym.JOB.txt || exit 1;
  
  # Get soft counts of all pronunciations from arc level information.
  cat $dir/lats/arc_info_sym.*.txt | steps/dict/get_pron_stats.py - \
    $dir/phone_decode/phone_map.txt $dir/lats/pron_stats.txt || exit 1;
fi

if [ $stage -le 7 ]; then
  echo "$0: Prune the pronunciation candidates generated from G2P/phone-decoding, and re-do lattice-alignment."
  mkdir -p $dir/dict_combined2
  cp $ref_dict/{extra_questions.txt,optional_silence.txt,nonsilence_phones.txt,silence_phones.txt} \
    $dir/dict_combined2/ 2>/dev/null
  rm $dir/dict_combined2/lexiconp.txt 2>/dev/null

  steps/dict/prune_pron_candidates.py $dir/lats/pron_stats.txt $dir/ref_lexicon.txt $dir/pruned_prons.txt
 
  awk 'NR==FNR{a[$0] = 1; next} (!($0 in a))' $dir/pruned_prons.txt $dir/phone_decoding_generated_prons.txt \
    > $dir/phone_decoding_generated_prons_pruned.txt

  awk 'NR==FNR{a[$0] = 1; next} (!($0 in a))' $dir/pruned_prons.txt $dir/g2p_generated_prons.txt \
    > $dir/g2p_generated_prons_pruned.txt \

  cat $dir/phone_decoding_generated_prons_pruned.txt $dir/g2p_generated_prons_pruned.txt \
    $dir/ref_lexicon.txt | tr -s '\t' ' ' | \
    awk 'NR==FNR{a[$1] = 1; next} ($1 in a)' $dir/train_counts.txt - | \
    cat $dir/non_scored_entries - | \
    sort | uniq > $dir/dict_combined2/lexicon.txt

  utils/prepare_lang.sh --phone-symbol-table $ref_lang/phones.txt \
    $dir/dict_combined2 $oov_symbol \
    $dir/lang_combined2_local $dir/lang_combined2 || exit 1;
  
  mdl=$src_mdl && $retrain_src_mdl && mdl=${src_mdl}_${affix}
  steps/align_fmllr_lats.sh --cmd "$decode_cmd" --nj $nj \
    $data $dir/lang_combined2 $mdl $dir/lats2 || exit 1;

  # Get arc level information from the lattice.
  $cmd JOB=1:$nj $dir/lats2/log/get_arc_info.JOB.log \
    lattice-align-words $dir/lang_combined2/phones/word_boundary.int \
    $dir/lats2/final.mdl \
    "ark:gunzip -c $dir/lats2/lat.JOB.gz |" ark:- \| \
    lattice-arc-post --acoustic-scale=0.1 $dir/lats2/final.mdl ark:- - \| \
    utils/int2sym.pl -f 5 $dir/lang_combined2/words.txt \| \
    utils/int2sym.pl -f 6- $dir/lang_combined2/phones.txt '>' \
    $dir/lats2/arc_info_sym.JOB.txt || exit 1;
  
  # Get soft counts of all pronunciations from arc level information.
  cat $dir/lats2/arc_info_sym.*.txt | steps/dict/get_pron_stats.py - \
    $dir/phone_decode/phone_map.txt $dir/lats2/pron_stats.txt || exit 1;
fi

if [ $stage -le 8 ]; then
  echo "$0: Select pronunciations according to the acoustic evidence from lattice alignment."
  # Given the acoustic evidence (soft-counts), we use a Bayesian framework to select pronunciations 
  # from three exclusive candidate sources: reference (hand-derived) lexicon, G2P and phone decoding.
  # The posteriors for all candidate prons for all words are printed into pron_posteriors.txt
  # For words which are out of the ref. vocab, the learned prons are written into out_of_ref_vocab_prons_learned.txt.
  # For words in the ref. vocab, we instead output a human readable & editable "edits" file called
  # ref_lexicon_edits.txt, which records all proposed changes to the prons (if any). Also, a 
  # summary is printed into the log file.
  
  variants_counts=`cat $dir/target_num_prons_per_word` || exit 1;
  $cmd $dir/lats2/log/select_prons_bayesian.log \
    steps/dict/select_prons_bayesian.py --prior-mean=$prior_mean --prior-counts-tot=$prior_counts_tot \
    --variants-counts=$variants_counts --variants-prob-mass=$variants_prob_mass --variants-prob-mass2=$variants_prob_mass2 \
    $ref_dict/silence_phones.txt $dir/lats2/pron_stats.txt $dir/train_counts.txt $dir/ref_lexicon.txt \
    $dir/g2p_generated_prons_pruned.txt $dir/phone_decoding_generated_prons_pruned.txt \
    $dir/lats2/pron_posteriors.temp $dir/lats2/out_of_ref_vocab_prons_learned.txt $dir/lats2/ref_lexicon_edits.txt

  # We reformat the pron_posterior file and add some comments.
  paste <(cat $dir/lats2/pron_posteriors.temp | cut -d' ' -f1-3 | column -t) \
    <(cat $dir/lats2/pron_posteriors.temp | cut -d' ' -f4-) | sort -nr -k1,3 | \
    cat <( echo ';; <word> <source: R(eference)/G(2P)/P(hone-decoding)> <posterior> <pronunciation>') -  \
    > $dir/lats2/pron_posteriors.txt
  rm $dir/pron_posteriors.temp 2>/dev/null
  # Remove some stuff that takes up space and are not likely to be visited by the user.
  if $cleanup; then
    rm -r $dir/lats2/{fsts*,lat*} 2>/dev/null
  fi
fi

if [ $stage -le 9 ]; then
  echo "$0: Expand the learned lexicon further to cover words in target vocab that are."
  echo "  ... not seen in acoustic training data."
  mkdir -p $dir/g2p/oov_learned_target_vocab
  cat $dir/lats2/out_of_ref_vocab_prons_learned.txt $dir/ref_lexicon.txt |
    awk 'NR==FNR{a[$1] = 1; next} !($1 in a)' - \
      $dir/target_vocab.txt | sort | uniq > $dir/out_of_target_vocab_words.txt
  
  variants_counts=`cat $dir/target_num_prons_per_word` || exit 1;
  steps/dict/apply_g2p.sh --var-counts $variants_counts --var-mass $variants_prob_mass \
    $dir/out_of_target_vocab_words.txt $dir/g2p $dir/g2p/oov_lex_target_vocab || exit 1;
  
  mkdir -p $dest_dict
  cp $ref_dict/{extra_questions.txt,optional_silence.txt,nonsilence_phones.txt,silence_phones.txt} \
    $dest_dict  2>/dev/null
  rm $dest_dict/lexiconp.txt 2>/dev/null

  cat $dir/g2p/oov_lex_target_vocab/lexicon.lex |  cut -f 1,3 -d$'\t' | \
    cat - $dir/lats2/out_of_ref_vocab_prons_learned.txt $dir/ref_lexicon.txt | \
    tr -s '\t' ' ' | sort | uniq > $dest_dict/lexicon.temp

  awk 'NR==FNR{a[$1] = 1; next} ($1 in a)' $dir/target_vocab.txt \
    $dest_dict/lexicon.temp | sort | uniq > $dest_dict/lexicon.nosil

  cat $dir/non_scored_entries $dest_dict/lexicon.nosil | sort | uniq >$dest_dict/lexicon0.txt
fi

if [ $stage -le 10 ]; then
  if [ $apply_edits ]; then
    echo "$0: Apply the ref_lexicon_edits file to the reference lexicon."
    cp $dir/lats2/ref_lexicon_edits.txt $dest_dict/lexicon_edits.txt 2>/dev/null
    steps/dict/apply_lexicon_edits.py $dest_dict/lexicon0.txt $dir/lats2/ref_lexicon_edits.txt - | \
      sort -u > $dest_dict/lexicon.txt
  else
    echo "$0: The user requested not to apply the ref_lexicon_edits file. So we just exit at this point."
    echo "  ... and leave it to the user to inspect/modify the edits file and then apply it on"
    echo "  ... $dest_dict/lexicon0.txt to produce the final learned lexicon $dest_dict/lexicon.txt."
    cp $dir/lats2/ref_lexicon_edits.txt $dest_dict/lexicon_edits.txt 2>/dev/null
  fi
fi
