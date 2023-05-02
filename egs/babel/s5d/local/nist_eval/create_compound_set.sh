#!/usr/bin/env bash

#Simple script to create compound set info that will allow for more automatized
#work with the shadow set.
#
#The notion of shadow data set came from the need to be able to verify
#the output of the recognizer during decoding the evaluation data.
#The idea is simple -- instead of decoding just the eval data, decode both
#eval data plus the dev data (or at least some portion of it) interleved
#randomly
#After decoding, we can isolate (split) the output from the decoding (and kws)
#so that we can score the dev data subset and if the score is identical to
#the score obtained by decoding the dev set previously, we can be little bit
#more sure that the eval set results are correct.

. ./path.sh

flen=0.01

[ ! -f lang.conf ] && echo "File lang.conf must exist (and contain a valid config)"
. ./lang.conf

devset=dev10h.pem
evlset=eval.seg
tgtset=shadow.seg
tgtdir=

. utils/parse_options.sh
[ -z $tgtdir ] && tgtdir=data/$tgtset

devset_basename=${devset%%.*}
devset_segments=${devset#*.}

evlset_basename=${evlset%%.*}
evlset_segments=${evlset#*.}

eval devset_flist=\$${devset_basename}_data_list
eval devset_ecf=\$${devset_basename}_ecf_file
eval devset_rttm=\$${devset_basename}_rttm_file
eval devset_stm=\$${devset_basename}_stm_file

eval evlset_flist=\$${evlset_basename}_data_list
eval evlset_ecf=\$${evlset_basename}_ecf_file
eval evlset_rttm=\$${evlset_basename}_rttm_file
eval evlset_stm=\$${evlset_basename}_stm_file

rm -rf $tgtdir/compounds
mkdir -p $tgtdir/compounds
mkdir -p $tgtdir/compounds/$devset
mkdir -p $tgtdir/compounds/$evlset

echo "Creating compound $tgtdir/compounds/$devset"
(
  echo "DEVSET file list: $devset_flist"
  cat `utils/make_absolute.sh $devset_flist` > $tgtdir/compounds/$devset/files.list
  echo "DEVSET ECF file : $devset_ecf"
  cat `utils/make_absolute.sh $devset_ecf` > $tgtdir/compounds/$devset/ecf.xml
  echo "DEVSET RTTM file: $devset_rttm"
  cat `utils/make_absolute.sh $devset_rttm` > $tgtdir/compounds/$devset/rttm
  echo "DEVSET STM file : $devset_stm"
  cat `utils/make_absolute.sh $devset_stm` | sed 's/ 1 / A /g' > $tgtdir/compounds/$devset/stm

  cat $tgtdir/segments | grep -w -F -f $tgtdir/compounds/$devset/files.list > $tgtdir/compounds/$devset/segments
  awk '{print $1}' $tgtdir/compounds/$devset/segments > $tgtdir/compounds/$devset/utterances

  for kwset_path in $tgtdir/kwset_*; do
    kwset=`basename $kwset_path`
    output=$tgtdir/compounds/$devset/$kwset

    mkdir -p $output/tmp
    cp $tgtdir/$kwset/kwlist.xml $output/
    cp $tgtdir/$kwset/utt.map $output/
    cp $tgtdir/compounds/$devset/ecf.xml $output/
    cp $tgtdir/compounds/$devset/rttm $output/
    local/search/rttm_to_hitlists.sh --segments $tgtdir/segments \
      --utt-table $tgtdir/$kwset/utt.map $tgtdir/compounds/$devset/rttm  \
      $tgtdir/$kwset/kwlist.xml  $tgtdir/compounds/$devset/ecf.xml \
      $output/tmp $output/hitlist 2> $output/hitlist.fails

    n1=`cat $output/hitlist.fails | wc -l`
    n2=`awk '{print $13}' $output/hitlist.fails | sort |uniq -c | wc -l`

    echo "INFO: For kwlist $kwset, $n2 KW types won't be found ($n1 tokens in total)"

    duration=$(cat $devset_ecf | perl -ne  'BEGIN{$dur=0;}{next unless $_ =~ /dur\=/; s/.*dur="([^"]*)".*/$1/; $dur+=$_;}END{print $dur/2}')

    echo $duration > $output/trials
    echo $flen > $output/frame_length

    echo "Number of trials: `cat $output/trials`"
    echo "Frame lengths: `cat $output/frame_length`"
    {
      cat $tgtdir/$kwset/f4de_attribs | grep kwlist_name
      language=$(grep kwlist  $tgtdir/$kwset/kwlist.xml | head -n 1 |  sed -E 's/.*language="([^"]*)".*/\1/g')
      echo "language=$language"
      echo "flen=$flen"
    } > $output/f4de_attribs

    cp $tgtdir/$kwset/categories $output/
  done
)

echo "Creating compound $tgtdir/compounds/$evlset"
(
  echo "EVLSET file list: $evlset_flist"
  cat `utils/make_absolute.sh $evlset_flist` > $tgtdir/compounds/$evlset/files.list
  echo "EVLSET ECF file : $evlset_ecf"
  cat `utils/make_absolute.sh $evlset_ecf` > $tgtdir/compounds/$evlset/ecf.xml
  if [ ! -z "$evlset_rttm" ]; then
    echo "EVLSET RTTM file: $evlset_rttm"
    cat `utils/make_absolute.sh $evlset_rttm` > $tgtdir/compounds/$evlset/rttm
  fi
  if [ ! -z "$evlset_stm" ]; then
    echo "EVLSET STM file : $evlset_stm"
    cat `utils/make_absolute.sh $evlset_stm` | sed 's/ 1 / A /g' > $tgtdir/compounds/$evlset/stm
  fi

  cat $tgtdir/segments | \
    grep -w -F -f $tgtdir/compounds/$evlset/files.list > $tgtdir/compounds/$evlset/segments
  awk '{print $1}' $tgtdir/compounds/$evlset/segments > $tgtdir/compounds/$evlset/utterances

  for kwset_path in $tgtdir/kwset_*; do
    kwset=`basename $kwset_path`
    output=$tgtdir/compounds/$evlset/$kwset

    mkdir -p $output/tmp
    cp $tgtdir/$kwset/kwlist.xml $output/
    cp $tgtdir/$kwset/utt.map $output/
    cp $tgtdir/compounds/$evlset/ecf.xml $output/

    if [  -f "$tgtdir/compounds/$evlset/rttm" ]; then
      cp $tgtdir/compounds/$evlset/rttm $output/
      local/search/rttm_to_hitlists.sh --segments $tgtdir/segments \
        --utt-table $tgtdir/$kwset/utt.map $tgtdir/compounds/$evlset/rttm  \
        $tgtdir/$kwset/kwlist.xml  $tgtdir/compounds/$evlset/ecf.xml \
        $output/tmp $output/hitlist 2> $output/hitlist.fails

      n1=`cat $output/hitlist.fails | wc -l`
      n2=`awk '{print $13}' $output/hitlist.fails | sort |uniq -c | wc -l`

      echo "INFO: For kwlist $kwset, $n2 KW types won't be found ($n1 tokens in total)"
    fi

    duration=$(cat $evlset_ecf | perl -ne  'BEGIN{$dur=0;}{next unless $_ =~ /dur\=/; s/.*dur="([^"]*)".*/$1/; $dur+=$_;}END{print $dur/2}')

    echo $duration > $output/trials
    echo $flen > $output/frame_length

    echo "Number of trials: `cat $output/trials`"
    echo "Frame lengths: `cat $output/frame_length`"
    {
      cat $tgtdir/$kwset/f4de_attribs | grep kwlist_name
      language=$(grep kwlist  $tgtdir/$kwset/kwlist.xml | head -n 1 |  sed -E 's/.*language="([^"]*)".*/\1/g')
      echo "language=$language"
      echo "flen=$flen"
    } > $output/f4de_attribs

    cp $tgtdir/$kwset/categories $output/
  done
)

echo "Compound creation OK."


