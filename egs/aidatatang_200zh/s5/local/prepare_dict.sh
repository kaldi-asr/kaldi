#!/usr/bin/env bash
#Copyright 2016 LeSpeech (Author: Xingyu Na)

# prepare dictionary for aidatatang
# it is done for English and Chinese separately,
# For English, we use CMU dictionary, and Sequitur G2P
# for OOVs, while all englist phone set will concert to Chinese
# phone set at the end. For Chinese, we use an online dictionary,
# for OOV, we just produce pronunciation using Charactrt Mapping.

. ./path.sh

[ $# != 0 ] && echo "Usage: $0" && exit 1;

train_dir=data/local/train
dev_dir=data/local/dev
test_dir=data/local/test
dict_dir=data/local/dict
mkdir -p $dict_dir
mkdir -p $dict_dir/lexicon-{en,ch}

# extract full vocabulary
cat $train_dir/text $dev_dir/text $test_dir/text | awk '{for (i = 2; i <= NF; i++) print $i}' |\
  perl -ape 's/ /\n/g;' | sort -u | grep -v '\[LAUGHTER\]' | grep -v '\[NOISE\]' |\
  grep -v '\[VOCALIZED-NOISE\]' > $dict_dir/words.txt || exit 1;

# split into English and Chinese
cat $dict_dir/words.txt | grep '[a-zA-Z]' > $dict_dir/lexicon-en/words-en.txt || exit 1;
cat $dict_dir/words.txt | grep -v '[a-zA-Z]' > $dict_dir/lexicon-ch/words-ch.txt || exit 1;


##### produce pronunciations for english
if [ ! -f $dict_dir/cmudict/cmudict.0.7a ]; then
  echo "--- Downloading CMU dictionary ..."
  svn co -r 13068 https://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict \
    $dict_dir/cmudict || exit 1;
fi

# format cmudict
echo "--- Striping stress and pronunciation variant markers from cmudict ..."
perl $dict_dir/cmudict/scripts/make_baseform.pl \
  $dict_dir/cmudict/cmudict.0.7a /dev/stdout |\
  sed -e 's:^\([^\s(]\+\)([0-9]\+)\(\s\+\)\(.*\):\1\2\3:' > $dict_dir/cmudict/cmudict-plain.txt || exit 1;

# extract in-vocab lexicon and oov words
echo "--- Searching for English OOV words ..."
awk 'NR==FNR{words[$1]; next;} !($1 in words)' \
  $dict_dir/cmudict/cmudict-plain.txt $dict_dir/lexicon-en/words-en.txt |\
  egrep -v '<.?s>' > $dict_dir/lexicon-en/words-en-oov.txt || exit 1;

awk 'NR==FNR{words[$1]; next;} ($1 in words)' \
  $dict_dir/lexicon-en/words-en.txt $dict_dir/cmudict/cmudict-plain.txt |\
  egrep -v '<.?s>' > $dict_dir/lexicon-en/lexicon-en-iv.txt || exit 1;

wc -l $dict_dir/lexicon-en/words-en-oov.txt
wc -l $dict_dir/lexicon-en/lexicon-en-iv.txt

# setup g2p and generate oov lexicon
if [ ! -f conf/g2p_model ]; then
  echo "--- Downloading a pre-trained Sequitur G2P model ..."
  wget http://sourceforge.net/projects/kaldi/files/sequitur-model4 -O conf/g2p_model
  if [ ! -f conf/g2p_model ]; then
    echo "Failed to download the g2p model!"
    exit 1
  fi
fi

echo "--- Preparing pronunciations for OOV words ..."
g2p=`which g2p.py`
if [ ! -x $g2p ]; then
  echo "g2p.py is not found. Checkout tools/extras/install_sequitur.sh."
  exit 1
fi
g2p.py --model=conf/g2p_model --apply $dict_dir/lexicon-en/words-en-oov.txt \
  > $dict_dir/lexicon-en/lexicon-en-oov.txt || exit 1;

# merge in-vocab and oov lexicon
cat $dict_dir/lexicon-en/lexicon-en-oov.txt $dict_dir/lexicon-en/lexicon-en-iv.txt |\
  sort > $dict_dir/lexicon-en/lexicon-en-phn.txt || exit 1;

# convert cmu phoneme to pinyin phonenme
mkdir -p $dict_dir/map
cat conf/cmu2pinyin | awk '{print $1;}' | sort -u > $dict_dir/map/cmu || exit 1;
cat conf/pinyin2cmu | awk -v cmu=$dict_dir/map/cmu \
  'BEGIN{while((getline<cmu)) dict[$1] = 1;}
   {for (i = 2; i <=NF; i++) if (dict[$i]) print $i;}' | sort -u > $dict_dir/map/cmu-used || exit 1;
cat $dict_dir/map/cmu | awk -v cmu=$dict_dir/map/cmu-used \
  'BEGIN{while((getline<cmu)) dict[$1] = 1;}
   {if (!dict[$1]) print $1;}' > $dict_dir/map/cmu-not-used || exit 1;

awk 'NR==FNR{words[$1]; next;} ($1 in words)' \
  $dict_dir/map/cmu-not-used conf/cmu2pinyin |\
  egrep -v '<.?s>' > $dict_dir/map/cmu-py || exit 1;

cat $dict_dir/map/cmu-py | \
  perl -e '
  open(MAPS, $ARGV[0]) or die("could not open map file");
  my %py2ph;
  foreach $line (<MAPS>) {
    @A = split(" ", $line);
    $py = shift(@A);
    $py2ph{$py} = [@A];
  }
  my @entry;
  while (<STDIN>) {
    @A = split(" ", $_);
    @entry = ();
    $W = shift(@A);
    push(@entry, $W);
    for($i = 0; $i < @A; $i++) { push(@entry, @{$py2ph{$A[$i]}}); }
    print "@entry";
    print "\n";
  }
' conf/pinyin2cmu > $dict_dir/map/cmu-cmu || exit 1;

cat $dict_dir/lexicon-en/lexicon-en-phn.txt | \
  perl -e '
  open(MAPS, $ARGV[0]) or die("could not open map file");
  my %py2ph;
  foreach $line (<MAPS>) {
    @A = split(" ", $line);
    $py = shift(@A);
    $py2ph{$py} = [@A];
  }
  my @entry;
  while (<STDIN>) {
    @A = split(" ", $_);
    @entry = ();
    $W = shift(@A);
    push(@entry, $W);
    for($i = 0; $i < @A; $i++) {
      if (exists $py2ph{$A[$i]}) { push(@entry, @{$py2ph{$A[$i]}}); }
      else {push(@entry, $A[$i])};
    }
    print "@entry";
    print "\n";
  }
' $dict_dir/map/cmu-cmu > $dict_dir/lexicon-en/lexicon-en.txt || exit 1;


##### produce pronunciations for chinese
if [ ! -f $dict_dir/cedict/cedict_1_0_ts_utf-8_mdbg.txt ]; then
  echo "------------- Downloading cedit dictionary ---------------"
  mkdir -p $dict_dir/cedict
  wget -P $dict_dir/cedict http://www.mdbg.net/chindict/export/cedict/cedict_1_0_ts_utf-8_mdbg.txt.gz
  gunzip $dict_dir/cedict/cedict_1_0_ts_utf-8_mdbg.txt.gz
fi

cat $dict_dir/cedict/cedict_1_0_ts_utf-8_mdbg.txt | grep -v '#' | awk -F '/' '{print $1}' |\
 perl -e '
  while (<STDIN>) {
    @A = split(" ", $_);
    print $A[1];
    for($n = 2; $n < @A; $n++) {
      $A[$n] =~ s:\[?([a-zA-Z0-9\:]+)\]?:$1:;
      $tmp = uc($A[$n]);
      print " $tmp";
    }
    print "\n";
  }
 ' | sort -k1 > $dict_dir/cedict/ch-dict.txt || exit 1;

echo "--- Searching for Chinese OOV words ..."
awk 'NR==FNR{words[$1]; next;} !($1 in words)' \
  $dict_dir/cedict/ch-dict.txt $dict_dir/lexicon-ch/words-ch.txt |\
  egrep -v '<.?s>' > $dict_dir/lexicon-ch/words-ch-oov.txt || exit 1;

awk 'NR==FNR{words[$1]; next;} ($1 in words)' \
  $dict_dir/lexicon-ch/words-ch.txt $dict_dir/cedict/ch-dict.txt |\
  egrep -v '<.?s>' > $dict_dir/lexicon-ch/lexicon-ch-iv.txt || exit 1;

wc -l $dict_dir/lexicon-ch/words-ch-oov.txt
wc -l $dict_dir/lexicon-ch/lexicon-ch-iv.txt


# validate Chinese dictionary and compose a char-based
# dictionary in order to get OOV pronunciations
cat $dict_dir/cedict/ch-dict.txt |\
  perl -e '
  use utf8;
  binmode(STDIN,":encoding(utf8)");
  binmode(STDOUT,":encoding(utf8)");
  while (<STDIN>) {
    @A = split(" ", $_);
    $word_len = length($A[0]);
    $proun_len = @A - 1 ;
    if ($word_len == $proun_len) {print $_;}
  }
  ' > $dict_dir/cedict/ch-dict-1.txt || exit 1;

# extract chars
cat $dict_dir/cedict/ch-dict-1.txt | awk '{print $1}' |\
  perl -e '
  use utf8;
  binmode(STDIN,":encoding(utf8)");
  binmode(STDOUT,":encoding(utf8)");  
  while (<STDIN>) {
    @A = split(" ", $_);
    @chars = split("", $A[0]);
    foreach (@chars) {
      print "$_\n";
    }
  }
  ' | grep -v '^$' > $dict_dir/lexicon-ch/ch-char.txt || exit 1;

# extract individual pinyins
cat $dict_dir/cedict/ch-dict-1.txt |\
  awk '{for(i=2; i<=NF; i++) print $i}' |\
  perl -ape 's/ /\n/g;' > $dict_dir/lexicon-ch/ch-char-pinyin.txt || exit 1;

# first make sure number of characters and pinyins
# are equal, so that a char-based dictionary can
# be composed.
nchars=`wc -l < $dict_dir/lexicon-ch/ch-char.txt`
npinyin=`wc -l < $dict_dir/lexicon-ch/ch-char-pinyin.txt`
if [ $nchars -ne $npinyin ]; then
  echo "Found $nchars chars and $npinyin pinyin. Please check!"
  exit 1
fi

paste $dict_dir/lexicon-ch/ch-char.txt $dict_dir/lexicon-ch/ch-char-pinyin.txt |\
  sort -u > $dict_dir/lexicon-ch/ch-char-dict.txt || exit 1;

# create a multiple pronunciation dictionary
cat $dict_dir/lexicon-ch/ch-char-dict.txt |\
  perl -e '
  my $prev = "";
  my $out_line = "";
  while (<STDIN>) {
    @A = split(" ", $_);
    $cur = $A[0];
    $cur_py = $A[1];
    #print length($prev);
    if (length($prev) == 0) { $out_line = $_; chomp($out_line);}
    if (length($prev)>0 && $cur ne $prev) { print $out_line; print "\n"; $out_line = $_; chomp($out_line);}
    if (length($prev)>0 && $cur eq $prev) { $out_line = $out_line."/"."$cur_py";}
    $prev = $cur;
  }
  print $out_line;
  ' >  $dict_dir/lexicon-ch/ch-char-dict-mp.txt || exit 1;

# get lexicon for Chinese OOV words
local/create_oov_char_lexicon.pl $dict_dir/lexicon-ch/ch-char-dict-mp.txt \
  $dict_dir/lexicon-ch/words-ch-oov.txt > $dict_dir/lexicon-ch/lexicon-ch-oov.txt || exit 1;

# seperate multiple prons for Chinese OOV lexicon
cat $dict_dir/lexicon-ch/lexicon-ch-oov.txt |\
  perl -e '
  my @entry;
  my @entry1;
  while (<STDIN>) {
    @A = split(" ", $_);
    @entry = ();
    push(@entry, $A[0]);
    for($i = 1; $i < @A; $i++ ) {
      @py = split("/", $A[$i]);
      @entry1 = @entry;
      @entry = ();
      for ($j = 0; $j < @entry1; $j++) {
        for ($k = 0; $k < @py; $k++) {
          $tmp = $entry1[$j]." ".$py[$k];
          push(@entry, $tmp);
        }
      }
    }
    for ($i = 0; $i < @entry; $i++) {
      print $entry[$i];
      print "\n";
    }
  }
  ' > $dict_dir/lexicon-ch/lexicon-ch-oov-mp.txt || exit 1;

# compose IV and OOV lexicons for Chinese
cat $dict_dir/lexicon-ch/lexicon-ch-oov-mp.txt $dict_dir/lexicon-ch/lexicon-ch-iv.txt |\
  awk '{if (NF > 1 && $2 ~ /[A-Za-z0-9]+/) print $0;}' > $dict_dir/lexicon-ch/lexicon-ch.txt || exit 1;

# convert Chinese pinyin to CMU format
cat $dict_dir/lexicon-ch/lexicon-ch.txt | sed -e 's/U:/V/g' | sed -e 's/ R\([0-9]\)/ ER\1/g'|\
  utils/pinyin_map.pl conf/pinyin2cmu > $dict_dir/lexicon-ch/lexicon-ch-cmu.txt || exit 1;

# combine English and Chinese lexicons
cat $dict_dir/lexicon-en/lexicon-en.txt $dict_dir/lexicon-ch/lexicon-ch-cmu.txt |\
  sort -u > $dict_dir/lexicon1.txt || exit 1;

cat $dict_dir/lexicon1.txt | awk '{ for(n=2;n<=NF;n++){ phones[$n] = 1; }} END{for (p in phones) print p;}'| \
  sort -u |\
  perl -e '
  my %ph_cl;
  while (<STDIN>) {
    $phone = $_;
    chomp($phone);
    chomp($_);
    $phone =~ s:([A-Z]+)[0-9]:$1:;
    if (exists $ph_cl{$phone}) { push(@{$ph_cl{$phone}}, $_)  }
    else { $ph_cl{$phone} = [$_]; }
  }
  foreach $key ( keys %ph_cl ) {
     print "@{ $ph_cl{$key} }\n"
  }
  ' | sort -k1 > $dict_dir/nonsilence_phones.txt  || exit 1;

( echo SIL; echo SPN; echo NSN; echo LAU ) > $dict_dir/silence_phones.txt

echo SIL > $dict_dir/optional_silence.txt

# No "extra questions" in the input to this setup, as we don't
# have stress or tone

cat $dict_dir/silence_phones.txt| awk '{printf("%s ", $1);} END{printf "\n";}' > $dict_dir/extra_questions.txt || exit 1;
cat $dict_dir/nonsilence_phones.txt | perl -e 'while(<>){ foreach $p (split(" ", $_)) {
  $p =~ m:^([^\d]+)(\d*)$: || die "Bad phone $_"; $q{$2} .= "$p "; } } foreach $l (values %q) {print "$l\n";}' \
 >> $dict_dir/extra_questions.txt || exit 1;

# Add to the lexicon the silences, noises etc.
(echo '!SIL SIL'; echo '[VOCALIZED-NOISE] SPN'; echo '[NOISE] NSN'; echo '[LAUGHTER] LAU';
 echo '<UNK> SPN' ) | \
 cat - $dict_dir/lexicon1.txt  > $dict_dir/lexicon.txt || exit 1;

echo "$0: aidatatang_200zh dict preparation succeeded"
exit 0;
