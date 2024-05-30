#!/usr/bin/env bash
# Copyright (c) 2015, Johns Hopkins University (Yenda Trmal <jtrmal@gmail.com>)
# License: Apache 2.0

# Begin configuration section.
flen=0.01
icu_transform="Any-Lower"
# End configuration section
set -e -o pipefail
set -o nounset                              # Treat unset variables as an error


if [ $# -eq 6 ]; then
  ecf=$1
  rttm=$2
  kwlist=$3
  data=$4
  lang=$5
  output=$6
elif  [ $# -eq 5 ]; then
  ecf=$1
  rttm=""
  kwlist=$2
  data=$3
  lang=$4
  output=$5
else
  echo >&2 "Incorrect number of script parameters!"
fi

mkdir -p $output
for f in $ecf $kwlist; do
  [ ! -f $f ] && echo "Mandatory file \"$f\" does not exist."
done


# The first way how to compute the duration produced numbers significantly
# dufferent from the numbers reported by F4DE. I'm leaving it here to document
# the fact that the signal_duration field is not the same number as the sum
# of the individual durations (dur field in each <excerpt>)
#duration=`head -n 1 $ecf | sed 's/.*signal_duration=\"\([0-9.][0-9.]*\)\".*/\1/g'`
#duration=`echo print $duration/2.0 | perl`

duration=$(cat $ecf | perl -ne  'BEGIN{$dur=0;}{next unless $_ =~ /dur\=/; s/.*dur="([^"]*)".*/$1/; $dur+=$_;}END{print $dur/2}')

echo $duration > $output/trials
echo $flen > $output/frame_length

echo "Number of trials: `cat $output/trials`"
echo "Frame lengths: `cat $output/frame_length`"

echo "Generating map files"
cat $data/segments | awk 'BEGIN{i=1}; {print $1, i; i+=1;}' > $output/utt.map
cat $data/wav.scp | awk 'BEGIN{i=1}; {print $1, i; i+=1;}' > $output/wav.map

#This does not work cp --no-preserve=all $ecf $output/ecf.xml
cat $ecf > $output/ecf.xml
cat $kwlist > $output/kwlist.xml
[ ! -z "$rttm" ] && cat $rttm  > $output/rttm

{
  echo "kwlist_name=`basename $kwlist`"
  language=$(grep kwlist $kwlist | head -n 1 |  sed -E 's/.*language="([^"]*)".*/\1/g')
  echo "language=$language"
  echo "flen=$flen"
} > $output/f4de_attribs

cat ${kwlist} | \
  perl -ne '{
    chomp;
    next unless (m/<kwtext>/ || m/kwid/);
    if ($_ =~ m/<kwtext>/) {
      s/.*<kwtext>(.*)<\/kwtext>.*/$1/g;
      die "Undefined format of the kwlist file!" unless defined $kwid;
      print $kwid . "\t" . $_ . "\n"; }
    else {
      s/.*kwid="(.*)".*/$1/g; $kwid=$_;};
    }' > $output/keywords.txt


command -v uconv >/dev/null 2>&1 || {
  echo >&2 "I require uconv but it's not installed. Use $KALDI_ROOT/tools/extras/install_icu.sh to install it (or use the system packager)";
  exit 1;
}

if [ -z "$icu_transform" ]; then
  cp $lang/words.txt $output/words.txt
else
  uconv -f utf8 -t utf8 -x "${icu_transform}" -o $output/words.txt $lang/words.txt
fi

if [ -z "$icu_transform" ]; then
  cat $output/keywords.txt
else
  paste <(cut -f 1  $output/keywords.txt ) \
        <(cut -f 2  $output/keywords.txt | \
          uconv -f utf8 -t utf8 -x "${icu_transform}" )
fi | local/kwords2indices.pl --map-oov 0  $output/words.txt |\
  sort -u > $output/keywords.int


echo "Generating categories"
{
  local/search/create_categories.pl $output/keywords.txt
  cat $output/keywords.int | perl -ane '
     if (grep (/^0$/, @F[1..$#F])) {print  "$F[0] OOV=1\n";}
     else { print "$F[0] OOV=0\n";}'
} | local/search/normalize_categories.pl > $output/categories

if [ ! -z "$rttm" ] && [ -f $rttm ] ; then
  local/search/rttm_to_hitlists.sh --segments $data/segments --utt-table $output/utt.map\
    $rttm $kwlist $ecf $output/tmp $output/hitlist
else
  echo "Not generating hitlist, scoring won't be possible"
fi
echo "Done"


