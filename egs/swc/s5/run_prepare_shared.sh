#!/bin/bash -u

. ./cmd.sh
. ./path.sh

# Path to SWC downloaded. This script cannot download the corpus
# automatically for you because it is compulsory to sign an 
# agreement before downloading.
case $(hostname -d) in
  minigrid.dcs.shef.ac.uk) SWC_DIR=/share/spandh.ami1/usr/yulan/sw/kaldi/dev/kaldi-trunk/egs/swc/swc ;; # Sheffield,
esac
# Or select manually,
# SWC_DIR=...

. utils/parse_options.sh


# You need to download the dict into folder $SRCDICTDIR manually
# from the SWC website "http://mini-vm20.dcs.shef.ac.uk/swc/LM.html",
mkdir -p data/local/dict 		
LMDIR=data/lang
# SRCDICT=/share/spandh.ami1/asr/dev/mtg/swc/lib/dicts/swc.v2.30k/30k-3gram-int-swc.swc.v2.30k-combilex.3g-int-hd.v2.dct
SRCDICTDIR=/share/spandh.ami1/usr/yulan/sw/kaldi/dev/kaldi-trunk/egs/swc/swc.lm/
cat $SRCDICTDIR/*.dct | sort -u | sed 's/\\//g' | grep -v '<' > tmp
echo '<unk> oov'  >> tmp	# OOV space holder
sort tmp > data/local/dict/lexicon.txt
rm tmp
gawk '{$1="";print $0}' data/local/dict/lexicon.txt | sed 's/ /\n/g' | sort -u | awk '{if (NF==1){print $0}}' |  grep -v sil | grep -v oov  >  data/local/dict/nonsilence_phones.txt
echo 'sil' > data/local/dict/optional_silence.txt
echo -e "sil\noov" > data/local/dict/silence_phones.txt 
utils/prepare_lang.sh data/local/dict "<unk>"  data/local/lang  $LMDIR


# You need to download the LM as $SRCLM manually from the SWC
# website "http://mini-vm20.dcs.shef.ac.uk/swc/LM.html",
SRCLM=/share/spandh.ami1/asr/dev/mtg/swc/lib/nets/swc.v2.30k/30k-4gram-int-swc.swc.v2.30k-combilex.4g-int.arpa.gz               
mkdir -p data/local/lm/
cp $SRCLM  data/local/lm/             
LMNAME=`echo $SRCLM | gawk 'BEGIN{FS="/"}{print $NF}' | sed 's/\.gz//g'`
echo $LMNAME > data/local/lm/final_lm

final_lm=`cat data/local/lm/final_lm`
LM=$final_lm.pr1-7
prune-lm --threshold=1e-7 data/local/lm/$final_lm.gz /dev/stdout | gzip -c > data/local/lm/$LM.gz


less data/local/lm/$LM.gz  |   arpa2fst - | fstprint | utils/eps2disambig.pl | utils/s2eps.pl | fstcompile --isymbols=$LMDIR/words.txt --osymbols=$LMDIR/words.txt --keep_isymbols=false --keep_osymbols=false | fstrmepsilon > $LMDIR/G.fst

ln -s ./lang  data/lang_$LM

# utils/format_lm.sh data/lang data/local/lm/$LM.gz data/local/dict/lexicon.txt data/lang_$LM




echo "Done"
exit 0


