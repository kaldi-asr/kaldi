#! /bin/bash

# Copyright  2015 Tokyo Institute of Technology (Authors: Takafumi Moriya and Takahiro Shinozaki)
#            2015 Mitsubishi Electric Research Laboratories (Author: Shinji Watanabe)
# Apache 2.0
# Acknowledgement  This work was supported by JSPS KAKENHI Grant Number 26280055.

if [ $# -ne 3 ]; then
  echo "Usage: "`basename $0`" <speech-dir> <transcription-dir> <csj-version>"
  echo "e.g., "`basename $0`" /database/NINJAL/CSJ data/csj-data usb (or dvd)"
  echo "See comments in the script for more details"
  exit 1
fi

resource=$1
outd=$2
csjv=$3

set -e # exit on error

case "$csjv" in
    "merl" ) SDB=sdb/ ; WAV=WAV/ ; disc=CSJ2004 ;; # Set SDB directory and WAV directory respectively.
    "usb" ) SDB=MORPH/SDB/ ; WAV=WAV/ ; disc="core noncore" ;; # Set SDB directory and WAV directory respectively.
    "dvd" ) num=dvd        ; SDB=           ; WAV=     ; disc=$num`seq -s " "$num 3 17| sed "s/ $num$//"` ;; # Set preserved format name to $num.
    *) echo "Input variable is usb or dvd only. $csjv is UNAVAILABLE VERSION." && exit 1;
esac

[ ! -e $resource ] && echo "Not exist CSJ or incorrect PATH." && exit 1;

if [ ! -e $outd/.done_make_trans ];then
(
    echo "Make Transcription and PATH of WAV file."
    mkdir -p $outd
    rm -f $outd/README.txt
    echo "Contents about generated directory and file
          ## About each directory
          {dvd3(dvd) or core(usb)}                         :Contain training data
          eval/                                            :Official evaluation data set ( *** Extract from dvd *** )
          excluded/                                        :Same speaker data including evaluation data (e.g. A01M0056) ( *** Extract from dvd *** )

          ## About each file
          {dvd3(dvd) or core(usb)}/A01F0055
                                   A01F0055-trans.text     :Transcriptions (utterances with R-tags are removed)
                                   A01F0055-wav.list       :Path about existing wav file
                                   A01F0055.4lex           :File for making lexicon" >$outd/README.txt

    # Make transcription file for each dvd and each lecture
    [ ! -x "`which nkf `" ]\
        && echo "This processing is need to prepare \"nkf\" command. Please retry after installing command \"nkf\"." && exit 1;

    for vol in $disc ;do
        mkdir -p $outd/$vol
        (
            if [ $csjv = "merl" ]; then
                ids=`ls $resource/$vol/$SDB | sed 's:.sdb::g' | sed 's/00README.txt//g'`
            else
                ids=`ls $resource/${SDB}$vol | sed 's:.sdb::g' | sed 's/00README.txt//g'`
            fi

            for id in $ids; do
                mkdir -p $outd/$vol/$id

                case "$csjv" in
                    "usb" ) TPATH="$resource/${SDB}$vol" ; WPATH="$resource/${WAV}$vol" ;;
                    "dvd" ) TPATH="$resource/$vol/$id"   ; WPATH="$resource/$vol/$id" ;;
                    "merl" ) TPATH="$resource/$vol/$SDB" ; WPATH="$resource/$vol/$WAV" ;;
                esac

                local/csj_make_trans/csj2kaldi4m.pl $TPATH/${id}.sdb  $outd/$vol/$id/${id}.4lex $outd/$vol/$id/${id}.4trn.t || exit 1;
                local/csj_make_trans/csjconnect.pl 0.5 10 $outd/$vol/$id/${id}.4trn.t $id > $outd/$vol/$id/${id}-trans.text || exit 1;
                rm $outd/$vol/$id/${id}.4trn.t

                if [ -e $WPATH/${id}-L.wav ]; then
                    find $WPATH -iname "${id}-[L,R].wav" >$outd/$vol/$id/${id}-wav.list
                else
                    find $WPATH -iname ${id}.wav >$outd/$vol/$id/${id}-wav.list || exit 1;
                fi

            done

            if [ -s $outd/$vol/$id/${id}-trans.text ] ;then
                echo -n >$outd/$vol/.done_$vol
                echo "Complete processing transcription data in $vol"
            else
                echo "Bad processing of making transcriptions part" && exit;
            fi
        )&
    done
    wait

    if [ -e $outd/$vol/.done_$vol ] ;then
        echo -n >$outd/.done_make_trans
        echo "Done!"
    else
        echo "Bad processing of making transcriptions part" && exit;
    fi
)
fi

## Exclude speech data given by test set speakers.
if [ ! -e $outd/.done_mv_eval_dup ]; then
(
    echo "Make evaluation set 1, 2, 3. And exclude speech data given by test set speakers."
    mkdir -p $outd/{\eval,excluded}
    mkdir -p $outd/eval/eval{1,2,3}

    # Exclude speaker ID
    A01M0056="S05M0613 R00M0187 D01M0019 D04M0056 D02M0028 D03M0017"


    # Evaluation set ID
    eval1="A01M0110 A01M0137 A01M0097 A04M0123 A04M0121 A04M0051 A03M0156 A03M0112 A03M0106 A05M0011"
    eval2="A01M0056 A03F0072 A02M0012 A03M0016 A06M0064 A06F0135 A01F0034 A01F0063 A01F0001 A01M0141"
    eval3="S00M0112 S00F0066 S00M0213 S00F0019 S00M0079 S01F0105 S00F0152 S00M0070 S00M0008 S00F0148"

    # Speech data given by test set speakers (e.g. eval2 : A01M0056)
    for list in $A01M0056 ; do
        find . -type d -name $list | xargs -i mv {} $outd/excluded
    done
    wait

    # Evaluation data
    for list in $eval1 $eval2 $eval3 ; do
        find . -type d -name $list | xargs -i mv {} $outd/eval
    done
    wait

    mv $outd/eval/{A01M0110,A01M0137,A01M0097,A04M0123,A04M0121,A04M0051,A03M0156,A03M0112,A03M0106,A05M0011} $outd/eval/eval1
    mv $outd/eval/{A01M0056,A03F0072,A02M0012,A03M0016,A06M0064,A06F0135,A01F0034,A01F0063,A01F0001,A01M0141} $outd/eval/eval2
    mv $outd/eval/{S00M0112,S00F0066,S00M0213,S00F0019,S00M0079,S01F0105,S00F0152,S00M0070,S00M0008,S00F0148} $outd/eval/eval3

    [ 10 -eq `ls $outd/eval/eval1 | wc -l` ] && echo -n >$outd/eval/.done_eval1
    [ 10 -eq `ls $outd/eval/eval2 | wc -l` ] && echo -n >$outd/eval/.done_eval2
    [ 10 -eq `ls $outd/eval/eval3 | wc -l` ] && echo -n >$outd/eval/.done_eval3
    if [ 3 -eq `ls -a $outd/eval | grep done_eval | wc -l` ] ;then
        echo -n >$outd/.done_mv_eval_dup
        echo "Done!"
    else
        echo "Bad processing of making evaluation set part" && exit;
    fi
    )
fi

## make lexicon.txt
if [ ! -e $outd/.done_make_lexicon ]; then
  echo "Make lexicon file."
  (
    lexicon=$outd/lexicon
    rm -f $outd/lexicon/lexicon.txt
    mkdir -p $lexicon
    cat $outd/*/*/*.4lex | grep -v "+ー" | grep -v "++" | grep -v "×" > $lexicon/lexicon.txt
    sort -u $lexicon/lexicon.txt > $lexicon/lexicon_htk.txt
    local/csj_make_trans/vocab2dic.pl -p local/csj_make_trans/kana2phone -e $lexicon/ERROR_v2d -o $lexicon/lexicon.txt $lexicon/lexicon_htk.txt
    cut -d'+' -f1,3- $lexicon/lexicon.txt >$lexicon/lexicon_htk.txt
    cut -f1,3- $lexicon/lexicon_htk.txt | perl -ape 's:\t: :g' >$lexicon/lexicon.txt

    if [ -s $lexicon/lexicon.txt ] ;then
      echo -n >$outd/.done_make_lexicon
      echo "Done!"
    else
      echo "Bad processing of making lexicon file" && exit;
    fi
  )
fi

[ ! 3 -le `ls -a $outd | grep done | wc -l` ] \
  && echo "ERROR : Processing is incorrect." && exit;

echo "Finish processing original CSJ data" && echo -n >$outd/.done_make_all
