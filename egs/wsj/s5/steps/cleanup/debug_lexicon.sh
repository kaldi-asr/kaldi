#!/bin/bash
# Copyright 2014  Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0

# this script gets some stats that will help you debug the lexicon.

# Begin configuration section.
stage=1
remove_stress=false
nj=10  # number of jobs for various decoding-type things that we run.
cmd=run.pl
alidir=
# End configuration section

echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh # source the path.
. parse_options.sh || exit 1;

if [ $# != 5 ]; then
   echo "usage: $0 <data-dir> <lang-dir> <src-dir> <src-dict> <dir>"
   echo "e.g.: $0 data/train data/lang exp/tri4b data/local/dict/lexicon.txt exp/debug_lexicon"
   echo "main options (for others, see top of script file)"
   echo "  --nj <nj>                                        # number of parallel jobs"
   echo "  --cmd <cmd>                                      # command to run jobs, e.g. run.pl,queue.pl"
   echo "  --stage <stage>                                  # use to control partial reruns."
   echo "  --remove-stress <true|false>                     # if true, remove stress before printing analysis"
   echo "                                                   # note: if you change this, you only have to rerun"
   echo "                                                   # from stage 10."
   echo "  --alidir <alignment-dir>                         # if supplied, training-data alignments and transforms"
   echo "                                                   # are obtained from here instead of being generated."
   exit 1;
fi

data=$1
lang=$2
src=$3
srcdict=$4
dir=$5

set -e

for f in $data/feats.scp $lang/phones.txt $src/final.mdl $srcdict; do
  [ ! -f $f ] && echo "$0: expected file $f to exist" && exit 1;
done

mkdir -p $dir
utils/lang/check_phones_compatible.sh $lang/phones.txt $srcdir/phones.txt
cp $lang/phones.txt $dir

if [ -z $alidir ]; then
  alidir=${src}_ali_$(basename $data)
  if [ $stage -le 1 ]; then
    steps/align_fmllr.sh --cmd "$cmd" --nj $nj $data $lang $src $alidir
  fi
fi

phone_lang=data/$(basename $lang)_phone_bg

if [ $stage -le 2 ]; then
  utils/lang/make_phone_bigram_lang.sh $lang $alidir $phone_lang
fi

if [ $stage -le 3 ]; then
  utils/mkgraph.sh $phone_lang $src $src/graph_phone_bg
fi

if [ $stage -le 4 ]; then
  steps/decode_si.sh --skip-scoring true \
    --cmd "$cmd" --nj $nj --transform-dir $alidir \
    --acwt 0.25 --beam 10.0 --lattice-beam 5.0 --max-active 2500 \
    $src/graph_phone_bg $data $src/decode_$(basename $data)_phone_bg
fi

if [ $stage -le 5 ]; then
  steps/get_train_ctm.sh --print-silence true --use-segments false \
     --cmd "$cmd" $data $lang $alidir
fi

if [ $stage -le 6 ]; then
  steps/get_ctm.sh --use-segments false --cmd "$cmd" --min-lmwt 3 --max-lmwt 8 \
     $data $phone_lang $src/decode_$(basename $data)_phone_bg
fi

if [ $stage -le 7 ]; then
  mkdir -p $dir
  # lmwt=4 corresponds to the scale we decoded at.
  cp $src/decode_$(basename $data)_phone_bg/score_4/$(basename $data).ctm $dir/phone.ctm

  cp $alidir/ctm $dir/word.ctm
fi

if [ $stage -le 8 ]; then
# we'll use 'sort' to do most of the heavy lifting when processing the data.
# suppose word.ctm has an entry like
# sw02054 A 213.32 0.24 and
# we'll convert it into two entries like this, with the start and end separately:
# sw02054-A 0021332 START and
# sw02054-A 0021356 END and
#
# and suppose phone.ctm has lines like
# sw02054 A 213.09 0.24 sil
# sw02054 A 213.33 0.13 ae_B
# we'll convert them into lines where the time is derived the midpoint of the phone, like
# sw02054 A 0021321 PHONE sil
# sw02054 A 0021340 PHONE ae_B
# and then we'll remove the optional-silence phones and, if needed, the word-boundary markers from
# the phones, to get just
# sw02054 A 0021340 PHONE ae
# then after sorting and merge-sorting the two ctm files we can easily
# work out for each word, what the phones were during that time.

  grep -v '<eps>' $phone_lang/phones.txt | awk '{print $1, $1}' | \
    sed 's/_B$//' | sed 's/_I$//' | sed 's/_E$//' | sed 's/_S$//' >$dir/phone_map.txt

  cat $dir/phone.ctm | utils/apply_map.pl -f 5 $dir/phone_map.txt > $dir/phone_text.ctm > $dir/phone_mapped.ctm

  export LC_ALL=C

  cat $dir/word.ctm  | awk '{printf("%s-%s %010.0f START %s\n", $1, $2, 1000*$3, $5); printf("%s-%s %010.0f END %s\n", $1, $2, 1000*($3+$4), $5);}' | \
    sort > $dir/word_processed.ctm

  # filter out those utteraces which only appea in phone_processed.ctm but not in word_processed.ctm
  cat $dir/phone_mapped.ctm | awk '{printf("%s-%s %010.0f PHONE %s\n", $1, $2, 1000*($3+(0.5*$4)), $5);}' | \
    awk 'NR==FNR{a[$1] = 1; next} {if($1 in a) print $0}' $dir/word_processed.ctm - \
    > $dir/phone_processed.ctm

  # merge-sort both ctm's
  sort -m $dir/word_processed.ctm $dir/phone_processed.ctm > $dir/combined.ctm
fi

  # after merge-sort of the two ctm's, we add <eps> to cover "deserted" phones due to precision limits, and then merge all consecutive <eps>'s. 
if [ $stage -le 9 ]; then
  awk '{print $1, $3, $4}' $dir/combined.ctm | \
     perl -e ' while (<>) { chop; @A = split(" ", $_); ($utt, $a,$b) = @A;
     if ($a eq "START") { $cur_word = $b; @phones = (); }
     if ($a eq "END") { print $utt, " ", $cur_word, " ", join(" ", @phones), "\n"; }
     if ($a eq "PHONE") { if ($prev eq "END") {print $utt, " ", "<eps>", " ", $b, "\n";} else {push @phones, $b;}} $prev = $a;} ' |\
     awk 'BEGIN{merge_prev=0;} {utt=$1;word=$2;pron=$3;for (i=4;i<=NF;i++) pron=pron" "$i;
     if (word_prev == "<eps>" && word == "<eps>" && utt_prev == utt) {merge=0;pron_prev=pron_prev" "pron;} else {merge=1;} 
     if(merge_prev==1) {print utt_prev, word_prev, pron_prev;};
     merge_prev=merge; utt_prev=utt; word_prev=word; pron_prev=pron;}
     END{if(merge_prev==1) {print utt_prev, word_prev, pron_prev;}}' > $dir/ctm_prons.txt
  
  steps/cleanup/internal/get_non_scored_words.py $lang > $dir/non_scored_words
  steps/cleanup/internal/get_pron_stats.py $dir/ctm_prons.txt $phone_lang/phones/silence.txt $phone_lang/phones/optional_silence.txt $dir/non_scored_words - | \
    sort -nr > $dir/prons.txt  
fi

if [ $stage -le 10 ]; then
  if $remove_stress; then
    perl -e 'while(<>) { @A=split(" ", $_); for ($n=1;$n<@A;$n++) { $A[$n] =~ s/[0-9]$//; } print join(" ", @A) . "\n"; } ' \
      <$srcdict >$dir/lexicon.txt
  else
    cp $srcdict $dir/lexicon.txt
  fi
  silphone=$(cat $phone_lang/phones/optional_silence.txt)
  echo "<eps> $silphone" >> $dir/lexicon.txt

  awk '{count[$2] += $1;} END {for (w in count){print w, count[w];}}' \
      <$dir/prons.txt >$dir/counts.txt



  cat $dir/prons.txt | \
    if $remove_stress; then
      perl -e 'while(<>) { @A=split(" ", $_); for ($n=1;$n<@A;$n++) { $A[$n] =~ s/[0-9]$//; } print join(" ", @A) . "\n"; } '
    else
      cat
    fi | perl -e '
     print ";; <count-of-this-pron> <rank-of-this-pron> <frequency-of-this-pron> CORRECT|INCORRECT <word> <pron>\n";
     open(D, "<$ARGV[0]") || die "opening dict file $ARGV[0]";
     # create a hash of all reference pronuncations, and for each word, record
     # a list of the prons, separated by " | ".
     while (<D>) {
        @A = split(" ", $_); $is_pron{join(" ",@A)} = 1;
        $w = shift @A;
        if (!defined $prons{$w}) { $prons{$w} = join(" ", @A); }
        else { $prons{$w} = $prons{$w} . " | " . join(" ", @A); }
     }
     open(C, "<$ARGV[1]") || die "opening counts file $ARGV[1];";
     while (<C>) { @A = split(" ", $_); $word_count{$A[0]} = $A[1]; }
     while (<STDIN>) { @A = split(" ", $_);
       $count = shift @A; $word = $A[0]; $freq = sprintf("%0.2f", $count / $word_count{$word});
       $rank = ++$wcount{$word}; # 1 if top observed pron of word, 2 if second...
       $str = (defined $is_pron{join(" ", @A)} ? "CORRECT" : "INCORRECT");
       shift @A;
       print "$count $rank $freq $str $word \"" . join(" ", @A) . "\", ref = \"$prons{$word}\"\n";
     } ' $dir/lexicon.txt $dir/counts.txt  >$dir/pron_info.txt

  grep -v '^;;' $dir/pron_info.txt | \
     awk '{ word=$5; count=$1; if (tot[word] == 0) { first_line[word] = $0; }
            corr[word] += ($4 == "CORRECT" ? count : 0); tot[word] += count; }
          END {for (w in tot) { printf("%s\t%s\t%s\t\t%s\n", tot[w], w, (corr[w]/tot[w]), first_line[w]); }} ' \
     | sort -k1 -nr | cat <( echo ';; <total-count-of-word> <word> <correct-proportion>      <first-corresponding-line-in-pron_info.txt>') - \
      > $dir/word_info.txt
fi

if [ $stage -le 11 ]; then
  echo "$0: some of the more interesting stuff in $dir/pron_info.txt follows."
  echo "# grep -w INCORRECT $dir/pron_info.txt  | grep -w 1 | head -n 20"

  grep -w INCORRECT $dir/pron_info.txt  | grep -w 1 | head -n 20

  echo "$0: here are some other interesting things.."
  echo "# grep -w INCORRECT $dir/pron_info.txt  | grep -w 1 | awk '\$3 > 0.4 && \$1 > 10' | head -n 20"
  grep -w INCORRECT $dir/pron_info.txt  | grep -w 1 | awk '$3 > 0.4 && $1 > 10' | head -n 20

  echo "$0: here are some high-frequency words whose reference pronunciations rarely show up."
  echo "# awk '\$3 < 0.1' $dir/word_info.txt  | head -n 20"
  awk '$3 < 0.1 || $1 == ";;"' $dir/word_info.txt  | head -n 20


fi
