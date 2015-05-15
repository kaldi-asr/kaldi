#! /bin/bash

# Copyright  2015 Tokyo Institute of Technology (Authors: Takafumi Moriya and Takahiro Shinozaki)
#            2015 Mitsubishi Electric Research Laboratories (Author: Shinji Watanabe)
# Apache 2.0
# Acknowledgement  This work was supported by JSPS KAKENHI Grant Number 26280055.

if [ $# -ne 2 ]; then
  echo "Usage: "`basename $0`" <speech-dir> <transcription-dir> "
  echo "See comments in the script for more details"
  exit 1
fi


resource=$1
outd=$2

csjext=./local/csj_make_trans/csj2kaldi4m.pl
csjconnect=./local/csj_make_trans/csjconnect.pl
k2phone=./local/csj_make_trans/kana2phone
vocab2dic=./local/csj_make_trans/vocab2dic.pl
reform=./local/csj_make_trans/reform.pl

if [ ! -d ./csj-data/dvd17 ];then
(
mkdir -p $outd
rm $outd/al_sent4lex.txt

cp ./local/csj_make_trans/overview_csj-data $outd/README.txt

# Make transcription file for each dvd and each lecture
[ ! -x "`which nkf `" ]\
    && echo "This processing is need to prepare \"nkf\" command. Please retry after installing command \"nkf\"." && exit 1;

for vol in dvd{3..17} ;do
    mkdir -p $outd/$vol

    for id in `ls $resource/$vol`;do
	mkdir -p $outd/$vol/${id}
	rm -r $outd/$vol/00README.txt
	
	    (
		nkf -e -d $resource/$vol/$id/${id}.sdb > $outd/$vol/${id}/sdb.tmp
		$csjext $outd/$vol/${id}/sdb.tmp  $outd/$vol/$id/${id}.4lex $outd/$vol/$id/${id}.4trn.t 
		
		$csjconnect 0.5 10 $outd/$vol/$id/${id}.4trn.t $id > $outd/$vol/$id/${id}-trans.text
		
		
		[ -z `grep $id local/csj_make_trans/testset` ]\
                  && cat $outd/$vol/$id/${id}.4lex >> $outd/al_sent4lex.txt
		
	    	rm $outd/$vol/$id/{${id}.4trn.t,sdb.tmp}
		
		if [ -e $resource/$vol/$id/${id}-L.wav ]; then
                    find $resource/$vol/$id -iname "${id}-[L,R].wav" >$outd/$vol/$id/${id}-wav.list
		else
                    find $resource/$vol/$id -iname ${id}.wav >$outd/$vol/$id/${id}-wav.list
		fi
		
		
	    ) 
    done
done
wait
)
fi

## make lexicon.txt
if [ ! -f ./csj-data/lexicon/lexicon.txt ]; then
    (
    mkdir -p $outd/lexicon
    sort $outd/al_sent4lex.txt >lex.tmp123
    uniq lex.tmp123 > lex.tmp456
    ${vocab2dic} -p $k2phone -o lex.tmp123 lex.tmp456
    $reform lex.tmp123 | sort | uniq > $outd/lexicon/lexicon.txt
    mv $outd/al_sent4lex.txt $outd/lexicon
    rm lex.tmp123 lex.tmp456 ERROR
    )
fi

## Exclude speech data given by test set speakers.
if [ ! -d ./csj-data/[eval,excluded] ]; then
(
    mkdir -p $outd/eval
    mkdir -p $outd/excluded
    mkdir -p $outd/eval/eval1
    mkdir -p $outd/eval/eval2
    mkdir -p $outd/eval/eval3
    
    # Speech data given by test set speakers (eval2 : A01M0056)
    rm dup_list
    for line in `cat local/csj_make_trans/A01M0056_duplication | less`; do
	find $outd/dvd* -iname $line >>dup_list
    done
    for list in `cat dup_list | less`;do
	mv $list $outd/excluded
	cp dup_list $outd/excluded/duplication.list
    done
    wait
    
    # Evaluation data
    rm dup_list
    for line in `cat local/csj_make_trans/testset | less`; do
	find $outd/dvd* -iname $line >>dup_list
    done
    for list in `cat dup_list | less`;do
	mv $list $outd/eval
	cp dup_list $outd/eval/evaluation.list
    done
    wait
    
    rm dup_list
    
    mv $outd/eval/{A01M0110,A01M0137,A01M0097,A04M0123,A04M0121,A04M0051,A03M0156,A03M0112,A03M0106,A05M0011} $outd/eval/eval1
    mv $outd/eval/{A01M0056,A03F0072,A02M0012,A03M0016,A06M0064,A06F0135,A01F0034,A01F0063,A01F0001,A01M0141} $outd/eval/eval2
    mv $outd/eval/{S00M0112,S00F0066,S00M0213,S00F0019,S00M0079,S01F0105,S00F0152,S00M0070,S00M0008,S00F0148} $outd/eval/eval3
    )
fi

comp_num=`ls -l $outd | wc -l`
[ ! $comp_num -eq 20 ] \
  && echo "ERROR : Processing is incorrect." && exit 1;

echo "Finish processing original CSJ data"
