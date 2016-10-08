#prepares dictionary from NST documentation http://www.nb.no/sbfil/dok/nst_leksdat_se.pdf
#phone file(s) must be retrieved elsewhere


KALDI_ROOT=$(pwd)/../../..
dir=data/local/dict
mkdir -p $dir

if [ ! -f $dir/sv.leksikon.tar.gz ]; then 
    ( wget http://www.nb.no/sbfil/leksikalske_databaser/leksikon/sv.leksikon.tar.gz --directory-prefix=$dir)
fi
wait 

tar -xzf $dir/sv.leksikon.tar.gz -C $dir

if [ ! -f $dir/lexicon.txt ]; then
	( python3 local/parse_dict.py $dir/NST\ svensk\ leksikon/swe030224NST.pron/swe030224NST.pron $dir/lexicon_first.txt) #$dir/phones.txt)
fi

#remove duplicate lines
awk '!seen[$0]++' $dir/lexicon_first.txt > $dir/lexicon.txt
rm $dir/lexicon_first.txt


echo "Dictionary preparations done."