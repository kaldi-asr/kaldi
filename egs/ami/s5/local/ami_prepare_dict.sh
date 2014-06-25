#!/bin/bash

# Formatting the Mississippi State dictionary for use in Edinburgh. Differs 
# from the one in Kaldi s5 recipe in that it uses lower-case --Arnab (Jan 2013)

# To be run from one directory above this script.

. path.sh

#check existing directories
[ $# != 0 ] && echo "Usage: local/ami_ihm_data_prep_edin.sh" && exit 1;

sdir=data/local/annotations
wdir=data/local/dict
cmuurl=http://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict/
cmuver=cmudict.0.7a

req="$sdir/transcripts2 local/wordlist.50k"
[ ! -f "$sdir/transcripts2" ] && echo "No such file $sdir/transcripts2 (need to run ami_text_prep.sh first)" && exit 1;

mkdir -p $wdir

if [ ! -f $wdir/$cmuver ]; then
  wget -O $wdir/$cmuver svn $cmuurl/$cmuver
  wget -O $wdir/$cmuver.phones svn $cmuurl/$cmuver.phones
  wget -O $wdir/$cmuver.symbols svn $cmuurl/$cmuver.symbols
fi

grep -e "^;;;" -v $wdir/$cmuver | sort  > $dir/lexicon1.txt

cat $dir/lexicon1.txt | awk '{ for(n=2;n<=NF;n++){ phones[$n] = 1; }} END{for (p in phones) print p;}' | \
  grep -v sil > $dir/nonsilence_phones.txt  || exit 1;

( echo sil; echo spn; echo nsn; echo lau ) > $dir/silence_phones.txt

echo sil > $dir/optional_silence.txt

# No "extra questions" in the input to this setup, as we don't
# have stress or tone.
echo -n >$dir/extra_questions.txt

# Add to the lexicon the silences, noises etc.
( echo '!sil sil'; echo '[vocalized-noise] spn'; echo '[noise] nsn'; \
  echo '[laughter] lau'; echo '<unk> spn' ) \
  | cat - $dir/lexicon1.txt  > $dir/lexicon2.txt || exit 1;

pushd $wdir >&/dev/null
ln -sf lexicon2.txt lexicon.txt # This is the final lexicon.
popd >&/dev/null

echo Prepared input dictionary and phone-sets for AMI phase 1.

