#!/usr/bin/env bash
#

# To be run from one directory above this script.

## The input is some directory containing the switchboard-1 release 2
## corpus (LDC97S62).  Note: we don't make many assumptions about how
## you unpacked this.  We are just doing a "find" command to locate
## the .sph files.

# for example /mnt/matylda2/data/SWITCHBOARD_1R2

. ./path.sh

# The parts of the output of this that will be needed are
# [in data/local/dict/ ]
# lexicon.txt
# extra_questions.txt
# nonsilence_phones.txt
# optional_silence.txt
# silence_phones.txt


#check existing directories
[ $# != 0 ] && echo "Usage: local/fisher_prepare_dict.sh" && exit 1;

dir=data/local/dict
mkdir -p $dir
echo "Getting CMU dictionary"
svn co  https://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict  $dir/cmudict

# silence phones, one per line.
for w in sil laughter noise oov; do echo $w; done > $dir/silence_phones.txt
echo sil > $dir/optional_silence.txt

# For this setup we're discarding stress.
cat $dir/cmudict/cmudict.0.7a.symbols | sed s/[0-9]//g | \
 tr '[A-Z]' '[a-z]' | perl -ane 's:\r::; print;' | sort | uniq > $dir/nonsilence_phones.txt

# An extra question will be added by including the silence phones in one class.
cat $dir/silence_phones.txt| awk '{printf("%s ", $1);} END{printf "\n";}' > $dir/extra_questions.txt || exit 1;

grep -v ';;;' $dir/cmudict/cmudict.0.7a |  tr '[A-Z]' '[a-z]' | \
 perl -ane 'if(!m:^;;;:){ s:(\S+)\(\d+\) :$1 :; s:  : :; print; }' | \
 perl -ane '@A = split(" ", $_); for ($n = 1; $n<@A;$n++) { $A[$n] =~ s/[0-9]//g; } print join(" ", @A) . "\n";' | \
 sort | uniq > $dir/lexicon1_raw_nosil.txt || exit 1;

# Add prons for laughter, noise, oov
for w in `grep -v sil $dir/silence_phones.txt`; do
  echo "[$w] $w"
done | cat - $dir/lexicon1_raw_nosil.txt  > $dir/lexicon2_raw.txt || exit 1;


# This is just for diagnostics:
cat data/train_all/text  | \
  awk '{for (n=2;n<=NF;n++){ count[$n]++; } } END { for(n in count) { print count[n], n; }}' | \
  sort -nr > $dir/word_counts

cat $dir/word_counts | awk '{print $2}' > $dir/word_list

# between lexicon2_raw and lexicon3_expand we limit it to the words seen in
# the Fisher data.
utils/filter_scp.pl $dir/word_list $dir/lexicon2_raw.txt > $dir/lexicon3_expand.txt

# From lexicon2_raw to lexicon3_expand, we also expand the vocab for acronyms
# like c._n._n. and other underscore-containing things as long as the new vocab
# could be divided into finite parts contained in lexicon2_raw
cat $dir/lexicon2_raw.txt | \
  perl -e 'while(<STDIN>) { @A=split; $w = shift @A; $pron{$w} = join(" ", @A); }
     ($w) = @ARGV;  open(W, "<$w") || die "Error opening word-counts from $w";
     while(<W>) { # reading in words we saw in training data..
       ($c, $w) = split;
       if (!defined $pron{$w}) {
         @A = split("_", $w);
         if (@A > 1) {
           $this_pron = "";
           $pron_ok = 1;
           foreach $a (@A) {
             if (defined($pron{$a})) { $this_pron = $this_pron . "$pron{$a} "; }
             else { $pron_ok = 0; print STDERR "Not handling word $w, count is $c\n"; last; }
           }
           if ($pron_ok) { $new_pron{$w} = $this_pron;   }
         }
       }
     }
     foreach $w (keys %new_pron) { print "$w $new_pron{$w}\n"; }' \
   $dir/word_counts >> $dir/lexicon3_expand.txt || exit 1;


cat $dir/lexicon3_expand.txt  \
   <( echo "mm m"
      echo "<unk> oov" ) > $dir/lexicon4_extra.txt


cp $dir/lexicon4_extra.txt $dir/lexicon.txt
rm $dir/lexiconp.txt 2>/dev/null; # can confuse later script if this exists.

awk '{print $1}' $dir/lexicon.txt | \
  perl -e '($word_counts)=@ARGV;
   open(W, "<$word_counts")||die "opening word-counts $word_counts";
   while(<STDIN>) { chop; $seen{$_}=1; }
   while(<W>) {
     ($c,$w) = split;
     if (!defined $seen{$w}) { print; }
   } ' $dir/word_counts > $dir/oov_counts.txt

echo "*Highest-count OOVs are:"
head -n 20 $dir/oov_counts.txt

utils/validate_dict_dir.pl $dir
exit 0;



srcdir=data/local/train # This is where we downloaded some stuff..
dir=data/local/dict
mkdir -p $dir
srcdict=$srcdir/swb_ms98_transcriptions/sw-ms98-dict.text

# assume swbd_p1_data_prep.sh was done already.
[ ! -f "$srcdict" ] && echo "No such file $srcdict" && exit 1;

#(2a) Dictionary preparation:
# Pre-processing (Upper-case, remove comments)
grep -v '^#' $srcdict | tr '[a-z]' '[A-Z]' | awk 'NF>0' | sort > $dir/lexicon1.txt || exit 1;

cat $dir/lexicon1.txt | awk '{ for(n=2;n<=NF;n++){ phones[$n] = 1; }} END{for (p in phones) print p;}' | \
  grep -v SIL > $dir/nonsilence_phones.txt  || exit 1;

( echo SIL; echo SPN; echo NSN; echo LAU ) > $dir/silence_phones.txt

echo SIL > $dir/optional_silence.txt

# No "extra questions" in the input to this setup, as we don't
# have stress or tone.
echo -n >$dir/extra_questions.txt

# Add to the lexicon the silences, noises etc.
(echo '!SIL SIL'; echo '[VOCALIZED-NOISE] SPN'; echo '[NOISE] NSN'; echo '[LAUGHTER] LAU';
 echo '<UNK> SPN' ) | \
 cat - $dir/lexicon1.txt  > $dir/lexicon2.txt || exit 1;


# Map the words in the lexicon.  That is-- for each word in the lexicon, we map it
# to a new written form.  The transformations we do are:
# remove laughter markings, e.g.
# [LAUGHTER-STORY] -> STORY
# Remove partial-words, e.g.
# -[40]1K W AH N K EY
# becomes -1K
# and
# -[AN]Y IY
# becomes
# -Y
# -[A]B[OUT]- B
# becomes
# -B-
# Also, curly braces, which appear to be used for "nonstandard"
# words or non-words, are removed, e.g.
# {WOLMANIZED} W OW L M AX N AY Z D
# -> WOLMANIZED
# Also, mispronounced words, e.g.
#  [YEAM/YEAH] Y AE M
# are changed to just e.g. YEAM, i.e. the orthography
# of the mispronounced version.
# Note-- this is only really to be used in training.  The main practical
# reason is to avoid having tons of disambiguation symbols, which
# we otherwise would get because there are many partial words with
# the same phone sequences (most problematic: S).
# Also, map
# THEM_1 EH M -> THEM
# so that multiple pronunciations just have alternate entries
# in the lexicon.

local/swbd_map_words.pl -f 1 $dir/lexicon2.txt | sort | uniq > $dir/lexicon3.txt || exit 1;

cp $dir/lexicon3.txt $dir/lexicon.txt # This is the final lexicon.

echo Prepared input dictionary and phone-sets for Switchboard phase 1.
