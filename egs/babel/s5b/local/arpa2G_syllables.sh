# This is as arpa2G.sh but specialized for the per-syllable setup.  This is
# specific to the BABEL setup.
# The difference from arpa2G.sh is that (1) we have to change <unk> to <oov>, because
# <oov> is the name of the phone that was chosen to represent the unknown word [note:
# <unk> is special to SRILM, which is why it appears in the vocab]; and (2) we have
# a special step with fstrhocompose which we use to ensure that silence cannot appear
# twice in succession.  [Silence appears in the language model, which would naturally
# allow it to appear twice in succession.]

# input side, because <oov> is the name of the

lmfile=$1
langdir=$2
destdir=$3

mkdir -p $destdir;

# Make FST that we compose with to disallow >1 silence in a row.
last_id=`tail -n 1 $langdir/words.txt | awk '{print $2}'` || exit 1;
[ -z $last_id ] && echo Error getting silence-id from $langdir/words.txt && exit 1;
silence_id=`grep -w SIL $langdir/words.txt | awk '{print $2}'` || exit 1;
[ -z $silence_id ] && echo Error getting silence-id from $langdir/words.txt && exit 1;
rho=$[$last_id+1]

# state 0 is start-state.  state 1 is state after we saw silence.  state 2 is 
# "dead state/failure state" that is not coaccessible.
cat <<EOF | fstcompile > $destdir/rho.fst
0 1 $silence_id $silence_id
0 0 $rho $rho
1 2 $silence_id $silence_id
1 0 $rho $rho
0
1
EOF


gunzip -c $lmfile | \
    grep -v '<s> <s>' | grep -v '</s> <s>' |  grep -v '</s> </s>' | \
    sed 's/<unk>/<oov>/g' | \
    arpa2fst - | \
    fstprint | \
    utils/eps2disambig.pl | \
    utils/s2eps.pl | \
    fstcompile --isymbols=$langdir/words.txt \
    --osymbols=$langdir/words.txt  --keep_isymbols=false --keep_osymbols=false | \
    fstrhocompose "$rho" - $destdir/rho.fst | \
    fstrmepsilon > $destdir/G.fst || exit 1

fstisstochastic $destdir/G.fst || true

rm $destdir/rho.fst

exit 0
