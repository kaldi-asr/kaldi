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
# alternative pronounciations for all words. Then we combine these three
# exclusive sources of pronounciations: the source lexicon (supposedly 
# hand-derived), lexicons from G2P/phone decoding into one lexicon and then run 
# lattice alignment on the same data, to collect acoustic evidence (soft
# counts) of all pronounciations. Based on these statistics, and
# user-specified prior-counts for the three sources, we then use a Bayesian
# framework to compute posteriors of all pronounciations for each word,
# and then select best pronounciations for each word (up to target-num-prons-per-word
# variants, or at most produce a certain amount (var-mass) of posterior mass.
# target-num-prons-per-word is estimated from the reference lexicon.
set -x
stage=0

# Begin configuration section.  
cmd=run.pl
nj=
stage=6
oov_symbol=
min_prob=0.3
var_mass=0.7
prior_counts="1-0.6-0.4"
g2p_for_iv=true
affix="lex"
num_gauss=
num_leaves=
retrain_src_mdl=true
cleanup=true
# End configuration section.  

. ./path.sh
. utils/parse_options.sh

if [ $# -ne 6 ]; then
  echo "Usage: $0 [options] <ref-dict> <target-vocab> <data> <src-mdl> \\"
  echo "          <ref-lang> <dst-dict>."
  echo "  This script does lexicon expansion using a combination of acoustic"
  echo "  evidence and G2P to produce a lexicon that covers words of a target vocab:"
  echo ""               
  echo "Arguments:"
  echo " <ref-dict>     the dir which contains the reference lexicon (most probably hand-derived)"
  echo "                we want to expand/improve, and nonsilence_phones.txt,.etc which we need " 
  echo "                for building new dict dirs."
  echo " <target-vocab> the vocabulary we want the final learned lexicon to cover."
  echo " <data>         acoustic training data we use to get alternative"
  echo "                pronounciations and collet acoustic evidence."
  echo " <src-mdl>      the acoustic model based on which we re-train an SAT model" 
  echo "                to do phone level decoding (to get alternative pronounciations)"
  echo "                and lattice-alignment (to collect acoustic evidence for"
  echo "                evaluating all prounciations"
  echo " <ref-lang>     the reference lang dir which we use to get non-scored-words"
  echo "                like <UNK> for building new dict dirs"
  echo " <dst-dict>     the dict dir where we put the final learned lexicon, whose"
  echo "                vocab matches <target-vocab>"
  echo ""
  echo "Note: <target-vocab> and the vocab of <data> don't have to match. For words"
  echo "     who are in <target-vocab> but not seen in <data>, their pronounciations" 
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
  echo "  --min-prob <float>           # the cut-off parameter used to select alternative pronounciations from phone"
  echo "                               # decoding. A smaller min-prob means more candidates will be included."
  echo "  --prior-counts               # prior counts assigned to three exclusive pronounciations sources: "
  echo "         <float-float-float>   # reference lexicon, g2p, and phone decoding (used in the final Bayesian"
  echo "                               # pronounciation selection procedure). We recommend setting a larger prior"
  echo "                               # count for the reference lexicon, and the three counts should sum up to"
  echo "                               # 3 to 6 (may need tuning). e.g. '2-0.6-0.4'"
  echo "  --var-mass <float>           # In the Bayesian pronounciation selection procedure, for each word, after"
  echo "                               # computing posteriors for all candidate pronounciations, we select so"
  echo "                               # many variants of prons to produce at most this (var-mass) amount of "
  echo "                               # posterior mass. It's used when we apply G2P in a similiar fashion."
  echo "                               # A lower value is recommended (like 0.7) for a language whose average"
  echo "                               # pron variants per word is low, like ~2 for English."
  echo "  --g2p-for-iv <true|false>    # apply G2P for in-vocab words to get more alternative pronounciations."
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
    $target_vocab | sort | uniq > $dir/target_vocab
    
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
    $dir/target_vocab | sort | uniq > $dir/oov_target_vocab.txt
  
  variants_counts=`cat $dir/target_num_prons_per_word` || exit 1;
  steps/dict/apply_g2p.sh --var-counts $variants_counts --var-mass $var_mass $dir/oov_target_vocab.txt \
    $dir/g2p $dir/g2p/oov_lex_target_vocab || exit 1;
  
  mkdir -p $dir/dict_g2p_target_vocab
  cp $ref_dict/{extra_questions.txt,optional_silence.txt,nonsilence_phones.txt,silence_phones.txt} \
    $dir/dict_g2p_target_vocab  2>/dev/null
  rm $dir/dict_g2p_target_vocab/lexiconp.txt 2>/dev/null

  cat $dir/g2p/oov_lex_target_vocab/lexicon.lex |  cut -f 1,3 -d$'\t' | \
    cat - $dir/dict_g2p_expanded_train/lexicon.txt | tr -s '\t' ' ' | sort | uniq \
    > $dir/dict_g2p_target_vocab/lexicon.temp

  awk 'NR==FNR{a[$1] = 1; next} ($1 in a)' $dir/target_vocab \
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
  echo "$0: Get alternative pronounciations from phone decoding results (for both oov and iv words), and from G2P for iv words (optional)"
  # Optionally, we apply G2P on in-vocab words to get more alternative pronounciations,
  # since we've found this helpful in some cases.
  if $g2p_for_iv; then
    echo "$0: Applying G2P on in-vocab words as requested."
    variants_counts=`cat $dir/target_num_prons_per_word` || exit 1;
    steps/dict/apply_g2p.sh --var-counts $[2*$variants_counts] $dir/ref_vocab.txt \
      $dir/g2p $dir/g2p/iv_lex || exit 1;
    cat $dir/g2p/iv_lex/lexicon.lex |  cut -f 1,3 -d$'\t' | cat - $dir/dict_g2p_expanded_train/lexicon.txt | \
      tr -s '\t' ' ' | awk 'NR==FNR{gsub(/\t/," ",$0); a[$0] = 1; next} {if(!($0 in a)) print $0}' \
      $dir/ref_lexicon.txt - | sort | uniq > $dir/g2p_generated_prons.txt || exit 1; 
  else
    cat $dir/dict_g2p_expanded_train/lexicon.txt | \
      awk 'NR==FNR{gsub(/\t/," ",$0); a[$0] = 1; next} {if(!($0 in a)) print $0}' $dir/ref_lexicon.txt - | \
      sort | uniq > $dir/g2p_generated_prons.txt || exit 1; 
  fi

  # We prune the phone-decoding generated prons relative to the largest count, by setting "min_prob",
  # and only leave prons who are not present in the reference lexicon / g2p-generated lexicon.
  cat $dir/ref_lexicon.txt $dir/g2p_generated_prons.txt > $dir/phone_decode/filter_lexicon.txt 
  $cmd $dir/phone_decode/log/prons_to_lexicon.log steps/dict/prons_to_lexicon.py \
    --min-prob=$min_prob --filter-lexicon=$dir/phone_decode/filter_lexicon.txt \
    $dir/phone_decode/prons.txt $dir/phone_decoding_generated_prons_unfiltered.txt
  cat $dir/phone_decoding_generated_prons_unfiltered.txt | grep -vP "<eps>|<UNK>|\[.*\]" | \
    sort | uniq > $dir/phone_decoding_generated_prons.txt || exit 1;
fi

if [ $stage -le 6 ]; then
  echo "$0: Combine the reference lexicon and pronounciations from phone-decoding/G2P into one"
  echo "  ... lexicon, and run lattice alignment using this lexicon on acoustic training data."
  
  mkdir -p $dir/dict_combined
  cp $ref_dict/{extra_questions.txt,optional_silence.txt,nonsilence_phones.txt,silence_phones.txt} \
    $dir/dict_combined/ 2>/dev/null
  rm $dir/dict_combined/lexiconp.txt 2>/dev/null

  cat $dir/phone_decoding_generated_prons.txt $dir/g2p_generated_prons.txt \
    $dir/ref_lexicon.txt | tr -s '\t' ' ' | sort | uniq > $dir/dict_combined/lexicon.txt

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
fi

if [ $stage -le 7 ]; then
  echo "$0: Collect acoustic evidence (soft counts) of all pronouncitions, and"
  echo "  ... select pronounciations according to the acoustic evidence."
  # Get soft counts of all pronounciations from arc level information.
  cat $dir/lats/arc_info_sym.*.txt | steps/dict/get_pron_stats.py - \
    $dir/phone_decode/phone_map.txt $dir/lats/pron_stats.txt || exit 1;
  
  # Given the acoustic evidence (soft-counts), we use a Bayesian framework to select pronounciations 
  # from three exclusive candidate sources: reference (hand-derived) lexicon, G2P and phone decoding.
  # The selection results are in lexicon_learned.txt. A summary is printed into the log file, and 
  # diagnostic info for "bad words" whose reference candidate prons were rejected (which may indicate
  # wrong reference prons / text normalization errors), is printed into diagnostic_info.txt, and 
  # posteriors for all prons is printed into pron_posteriors.txt
  variants_counts=`cat $dir/target_num_prons_per_word` || exit 1;
  $cmd $dir/lats/log/select_prons_bayesian.log \
    steps/dict/select_prons_bayesian.py --alpha=$prior_counts --variants-counts=$variants_counts \
    --variants-mass=$var_mass $dir/lats/pron_stats.txt $dir/train_counts.txt $dir/ref_lexicon.txt \
    $dir/g2p_generated_prons.txt $dir/phone_decoding_generated_prons.txt \
    $dir/lats/lexicon_learned.txt $dir/lats/pron_posteriors.temp $dir/lats/diagnostic_info.txt
   
  # We reformat the pron_posterior file and add some comments.
  paste <(cat $dir/lats/pron_posteriors.temp | cut -d' ' -f1-3 | column -t) \
    <(cat $dir/lats/pron_posteriors.temp | cut -d' ' -f4-) | sort -nr -k1,3 | \
    cat <( echo ';; <word> <source: R(eference)/G(2P)/P(hone-decoding)> <posterior> <pronounciation>') -  \
    > $dir/lats/pron_posteriors.txt
  rm $dir/pron_posteriors.temp 2>/dev/null
fi

if [ $stage -le 8 ]; then
  echo "$0: Expand the learned lexicon to cover words in target vocab that are."
  echo "  ... not seen in acoustic training data."
  mkdir -p $dir/g2p/oov_learned_target_vocab
  awk 'NR==FNR{a[$1] = 1; next} !($1 in a)' $dir/lats/lexicon_learned.txt \
    $dir/target_vocab | sort | uniq > $dir/oov_learned_target_vocab.txt
  
  variants_counts=`cat $dir/target_num_prons_per_word` || exit 1;
  steps/dict/apply_g2p.sh --var-counts $variants_counts --var-mass $var_mass \
    $dir/oov_learned_target_vocab.txt $dir/g2p $dir/g2p/oov_learned_lex_target_vocab || exit 1;
  
  mkdir -p $dest_dict
  cp $ref_dict/{extra_questions.txt,optional_silence.txt,nonsilence_phones.txt,silence_phones.txt} \
    $dest_dict  2>/dev/null
  rm $dest_dict/lexiconp.txt 2>/dev/null

  cat $dir/g2p/oov_learned_lex_target_vocab/lexicon.lex |  cut -f 1,3 -d$'\t' | \
    cat - $dir/lats/lexicon_learned.txt | tr -s '\t' ' ' | sort | uniq > $dest_dict/lexicon.temp

  awk 'NR==FNR{a[$1] = 1; next} ($1 in a)' $dir/target_vocab \
    $dest_dict/lexicon.temp | sort | uniq > $dest_dict/lexicon.nosil

  cat $dir/non_scored_entries $dest_dict/lexicon.nosil | sort | uniq >$dest_dict/lexicon.txt
fi

# Remove some stuff that takes up space and are not likely to be visited by the user.
if [ $stage -le 9 ] && $cleanup; then
  rm -r $dir/lang* $dir/lats/{fsts*,lat*} 2>/dev/null
fi
