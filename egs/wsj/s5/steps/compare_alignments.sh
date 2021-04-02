#!/usr/bin/env bash

# Copyright 2018  Johns Hopkins University (author: Daniel Povey)
# Apache 2.0.

set -e
stage=0
cmd=run.pl   # We use this only for get_ctm.sh, which can be a little slow.
num_to_sample=1000  # We sample this many utterances for human-readable display, starting from the worst and then
                    # starting from the middle.
cleanup=true

if [ -f ./path.sh ]; then . ./path.sh; fi

. ./utils/parse_options.sh

if [ $# -ne 5 ] && [ $# -ne 7 ]; then
  cat <<EOF
  This script compares two directories containing data alignments, and
  creates statistics showing how much the phone and word alignments differ,
  including breakdown by phones and words; and which utterances differ the
  most.  This is intended for diagnostic purposes.  Both alignment directories
  should be for the same data (or at least the data sets should overlap).
  The word alignment stats may not be correctly obtained if the data-dirs are
  not the same.

  Usage: $0 [options] <lang-directory> <data-directory> <ali-dir1> <ali-dir2> <work-dir>
    or:  $0 [options] <lang1> <lang2> <data1> <data2> <ali-dir1> <ali-dir2> <work-dir>
   e.g.: $0 data/lang data/train exp/tri2_ali exp/tri3_ali exp/compare_ali_2_3

  Options:
              --cmd (run.pl|queue.pl...)      # specify how to run the sub-processes.
                                              # (passed through to get_train_ctm.sh)
              --cleanup <true|false>          # Specify --cleanup false to prevent
                                              # cleanup of temporary files.
              --stage  <n>                    # Enables you to run part of the script.

EOF
  exit 1
fi

if [ $# -eq 5 ]; then
  lang1=$1
  lang2=$1
  data1=$2
  data2=$2
  ali_dir1=$3
  ali_dir2=$4
  dir=$5
else
  lang1=$1
  lang2=$2
  data1=$3
  data2=$4
  ali_dir1=$5
  ali_dir2=$6
  dir=$7
fi

for f in $lang1/phones.txt $lang2/phones.txt $data1/utt2spk $data2/utt2spk \
         $ali_dir1/ali.1.gz $ali_dir2/ali.2.gz; do
  if [ ! -f $f ]; then
    echo "$0: expected file $f to exist"
    exit 1
  fi
done

# This will exit if the phone symbol id's are different, due to
# `set -e` above.
utils/lang/check_phones_compatible.sh $lang1/phones.txt $lang2/phones.txt

nj1=$(cat $ali_dir1/num_jobs)
nj2=$(cat $ali_dir2/num_jobs)

mkdir -p $dir/log


if [ $stage -le 0 ]; then
  echo "$0: converting alignments to phones."

  for j in $(seq $nj1); do gunzip -c $ali_dir1/ali.$j.gz; done | \
    ali-to-phones --per-frame=true $ali_dir1/final.mdl ark:- ark:- | gzip -c > $dir/phones1.gz

  for j in $(seq $nj2); do gunzip -c $ali_dir2/ali.$j.gz; done | \
    ali-to-phones --per-frame=true $ali_dir2/final.mdl ark:- ark:- | gzip -c > $dir/phones2.gz
fi

if [ $stage -le 1 ]; then
  echo "$0: getting comparison stats and utterance stats."
  compare-int-vector --binary=false --write-confusion-matrix=$dir/conf.mat \
            "ark:gunzip -c $dir/phones1.gz|" "ark:gunzip -c $dir/phones2.gz|" 2>$dir/log/compare_phones.log > $dir/utt_stats.phones
  tail -n 8 $dir/log/compare_phones.log
fi

if [ $stage -le 3 ]; then
  cat $dir/conf.mat | grep -v -F '[' | sed 's/]//' | awk '{n=NF; for (k=1;k<=n;k++) { conf[NR,k] = $k; row_tot[NR] += $k; col_tot[k] += $k; } } END{
   for (row=1;row<=n;row++) for (col=1;col<=n;col++) {
     val = conf[row,col]; this_row_tot = row_tot[row]; this_col_tot = col_tot[col];
     rval=conf[col,row]
     min_tot = (this_row_tot < this_col_tot ? this_row_tot : this_col_tot);
     if (val != 0) {
       phone1 = row-1; phone2 = col-1;
       if (row == col) printf("COR %d %d %.2f%\n", phone1, val, (val * 100 / this_row_tot));
       else {
         norm_prob = val * val / min_tot;  # heuristic for sorting.
         printf("SUB %d %d %d %d %.2f%% %.2f%%\n",
                 norm_prob, phone1, phone2, val, (val * 100 / min_tot), (rval * 100 / min_tot)); }}}}' > $dir/phone_stats.all

   (
     echo "# Format: <phone> <frame-count> <percent-correct>"
     grep '^COR' $dir/phone_stats.all | sort -n -k4,4 | awk '{print $2, $3, $4}' | utils/int2sym.pl -f 1 $lang1/phones.txt
   ) > $dir/phones_correct.txt

   (
     echo "#Format: <phone1> <phone2> <num-frames> <prob-wrong%> <reverse-prob-wrong%>"
     echo "# <num-frames> is the number of frames that were labeled <phone1> in the first"
     echo "# set of alignments and <phone2> in the second."
     echo "# <prob-wrong> is <num-frames> divided by the smaller of the total num-frames of"
     echo "#  phone1 or phone2, whichever is smaller; expressed as a percentage."
     echo "#<reverse-prob-wrong> is the same but for the reverse substitution, from"
     echo "#<phone2> to <phone1>; the comparison with <prob-wrong> the substitutions are)."
     grep '^SUB' $dir/phone_stats.all | sort -nr -k2,2 | awk '{print $3,$4,$5,$6,$7}' | utils/int2sym.pl -f 1-2 $lang1/phones.txt
   ) > $dir/phone_subs.txt
fi

if [ $stage -le 4 ]; then
  echo "$0: getting CTMs"
  steps/get_train_ctm.sh --use-segments false --print-silence true --cmd "$cmd" --frame-shift 1.0 $data1 $lang1 $ali_dir1 $dir/ctm1
  steps/get_train_ctm.sh --use-segments false --print-silence true --cmd "$cmd" --frame-shift 1.0 $data2 $lang2 $ali_dir2 $dir/ctm2
fi

if [ $stage -le 5 ]; then
  oov=$(cat $lang1/oov.int)
  # Note: below, we use $lang1 for both setups; this is by design as compare-int-vector
  # assumes they use the same symbol table.
  for n in 1 2; do
    cat $dir/ctm${n}/ctm | utils/sym2int.pl --map-oov $oov -f 5 $lang1/words.txt | \
      awk 'BEGIN{utt_id="";} { if (utt_id != $1) { if (utt_id != "") printf("\n"); utt_id=$1; printf("%s ", utt_id); } t_start=int($3); t_end=t_start + int($4); word=$5; for (t=t_start; t<t_end; t++) printf("%s ", word); } END{printf("\n")}' | \
      copy-int-vector ark:- ark:- | gzip -c >$dir/words${n}.gz
  done
fi

if [ $stage -le 5 ]; then
  compare-int-vector --binary=false --write-tot-counts=$dir/words_tot.vec --write-diff-counts=$dir/words_diff.vec \
         "ark:gunzip -c $dir/words1.gz|" "ark:gunzip -c $dir/words2.gz|" 2>$dir/log/compare_words.log >$dir/utt_stats.words
  tail -n 8 $dir/log/compare_words.log
fi

if [ $stage -le 6 ]; then

  ( echo "# Word stats.  Format:";
    echo "<proportion-of-wrong-frames> <num-wrong-frames> <num-correct-frames> <word>"

    paste <(awk '{for (n=2;n<NF;n++) print $n;}' <$dir/words_diff.vec) \
      <(awk '{for (n=2;n<NF;n++) print $n;}' <$dir/words_tot.vec) | \
       awk '{ if($2 > 0) print $1*$1/$2, $1/$2, $1, $2, (NR-1)}' | utils/int2sym.pl -f 5 $lang1/words.txt | \
      sort -nr | awk '{print $2, $3, $4, $5;}'
  ) > $dir/word_stats.txt

fi

if [ $stage -le 7 ]; then
  for type in phones words; do
    num_utts=$(wc -l <$dir/utt_stats.$type)
    cat $dir/utt_stats.$type | awk -v type=$type 'BEGIN{print "Utterance-id proportion-"type"-changed num-frames num-wrong-frames"; }
          {print $1, $3 * 1.0 / $2, $2, $3; }' | sort -nr -k2,2 > $dir/utt_stats.$type.sorted
    (
      echo "$0: Percentiles 100, 90, .. 0 of proportion-$type-changed distribution (over utterances) are:"
    cat $dir/utt_stats.$type.sorted | awk -v n=$num_utts 'BEGIN{k=int((n-1)/10);} {if (NR % k == 1) printf("%s ", $2); } END{print "";}'
    ) | tee $dir/utt_stats.$type.percentiles
  done
fi


if [ $stage -le 8 ]; then
  # Display the 1000 worst utterances, and 1000 utterances from the middle of the pack, in a readable format.
  num_utts=$(wc -l <$dir/utt_stats.words.sorted)
  half_num_utts=$[$num_utts/2];
  if [ $num_to_sample -gt $half_num_utts ]; then
    num_to_sample=$half_num_utts
  fi
  head -n $num_to_sample $dir/utt_stats.words.sorted | awk '{print $1}' > $dir/utt_ids.worst
  tail -n +$half_num_utts $dir/utt_stats.words.sorted | head -n $num_to_sample | awk '{print $1}' > $dir/utt_ids.mid

  for suf in worst mid; do
    for n in 1 2; do
      gunzip -c $dir/phones${n}.gz | copy-int-vector ark:- ark,t:- | utils/filter_scp.pl $dir/utt_ids.$suf  >$dir/temp
      # the next command reorders them, and duplicates the utterance-idwhich we'll later use
      # that to display the word sequence.
      awk '{print $1,$1,$1}' <$dir/utt_ids.$suf | utils/apply_map.pl -f 3 $dir/temp > $dir/phones${n}.$suf
      rm $dir/temp
    done
    # the stuff with 0 and <eps> below is a kind of hack so that if the phones are the same, we end up
    # with just the phone, but if different, we end up with p1/p2.
    # The apply_map.pl stuff is to put the transcript there.

    (
      echo "# Format: <utterance-id> <word1> <word2> ... <wordN>  <frame1-phone> ... <frameN-phone>"
      echo "# If the two alignments have the same phone, just that phone will be printed;"
      echo "# otherwise the two phones will be printed, as in 'phone1/phone2'.  So '/' is present"
      echo "# whenever there is a mismatch."

      paste $dir/phones1.$suf $dir/phones2.$suf | perl -ane ' @A = split("\t", $_); @A1 = split(" ", $A[0]); @A2 = split(" ", $A[1]);
            $utt = shift @A1; shift @A2; print $utt, " ";
            for ($n = 0; $n < @A1 && $n < @A2; $n++) { $a1=$A1[$n]; $a2=$A2[$n];  if ($a1 eq $a2) { print "$a1 "; } else { print "$a1 0 $a2 "; }}
            print "\n" ' | utils/int2sym.pl -f 3- $lang1/phones.txt | sed 's: <eps> :/:g' | \
        utils/apply_map.pl -f 2 $data1/text
    )  > $dir/compare_phones_${suf}.txt
  done
fi


if [ $stage -le 9 ] && $cleanup; then
  rm $dir/phones{1,2}.gz $dir/words{1,2}.gz $dir/ctm*/ctm $dir/*.vec $dir/conf.mat \
     $dir/utt_ids.*  $dir/phones{1,2}.{mid,worst} $dir/utt_stats.{phones,words} \
     $dir/phone_stats.all
fi

# clean up
exit 0
