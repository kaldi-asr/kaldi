#!/usr/bin/env bash


help="Usage: $(basename $0) <input-lexicon-with-tabs> <word2syllable-lexicon-out> <syllable-lexicon-out>
    E.g. $(basename $0) data/local/lexicon.txt word2syllable_lexicon.txt data/local/syllables/lexicon.txt
    Here, <input-lexicon-with-tabs> is the text-form lexicon but with tabs separating the syllables, e.g.
    WORD  w o   rr d
    <word2syllable-lexicon-out> has entries of the form
    WORD  w/o  rr/d
    <syllable-lexicon-out> has entries of the form
    w/o  w o"

# config vars:
pron_probs=false # If you set --pron-probs true, will expect pron-prob on input lexicon and produce
                 # pron-probs on word2syllable lexicon.
# end configs.
. utils/parse_options.sh

if [ $# != 3 ]; then
  echo $help 2>&1;
  exit 1;
fi

lex_in=$1
w2s_lex_out=$2
s2p_lex_out=$3

[ ! -f $lex_in ] && echo "No such file $lex_in" && exit 1;
mkdir -p `dirname $w2s_lex_out`
mkdir -p `dirname $s2p_lex_out`

cat $lex_in | perl -e  '
  ($w2s, $pron_probs) = @ARGV;
  open(W2S, ">$w2s") || die "opening word to syllable lexicon";
  $saw_tabs = 0;
  while(<STDIN>) { 
    chop;
    if ($pron_probs eq "true") {
      m:(\S+)\s+(\S+)\s+(.+): || die "Bad line $_ (note: have pron probs).";
      $word = $1;
      $prob = $2;
      $pron = $3;
      ($prob > 0.0 && $prob <= 1.0) || die "Bad pron-prob $prob in line $_";
      print W2S "$word $prob";
    } else {
      m:(\S+)\s+(.+): || die "Bad line $_ (note: do not have pron probs).";
      $word = $1;
      $pron = $2;
      print W2S "$word";      
    }
    @A = split("\t", $pron);
    @A >= 1 || die "Bad lexicon line $_\n";
    if (@A > 1) { $saw_tabs = 1; }
    foreach $s (@A) {
      $s =~ s/^\s+//; # Remove leading space.
      $s =~ s/\s+$//; # Remove trailing space.
      if ($s ne "") {
        $s =~ m:/: && die "slash (/) present in syllable $s (not allowed)\n";
        $t = join("/", split(" ", $s)); # replace spaces with /
        print W2S " $t";
        print "$t $s\n";
      }
    }
    print W2S "\n";
  } 
  if (! $saw_tabs) {
    die "You seem to be using as input to this script, a lexicon that does not have " .
       "syllables separated by tabs.";
  }
  ' $w2s_lex_out $pron_probs | sort | uniq > $s2p_lex_out || exit 1;

exit 0;
