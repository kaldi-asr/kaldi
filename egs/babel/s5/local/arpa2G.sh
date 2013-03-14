#Simple utility script to convert the gziped ARPA lm into a G.fst file

lmfile=$1
langdir=$2
destdir=$3

gunzip -c $lmfile | \
    grep -v '<s> <s>' | grep -v '</s> <s>' |  grep -v '</s> </s>' | \
    arpa2fst - | \
    fstprint | \
    utils/eps2disambig.pl | \
    utils/s2eps.pl | \
    fstcompile --isymbols=$langdir/words.txt \
    --osymbols=$langdir/words.txt  --keep_isymbols=false --keep_osymbols=false | \
    fstrmepsilon > $destdir/G.fst || exit 1
fstisstochastic $destdir/G.fst 

exit 0
