. ./path.sh

min_lmwt=5
max_lmwt=25
cer=0
cmd=run.pl
. ./utils/parse_options.sh

min_lmwt_start=$min_lmwt
max_lmwt_start=$max_lmwt

datadir=$1; shift
name=$1; shift
. ./lang.conf

set -e
set -o pipefail

[ ! -d $datadir/compounds/$name ] && echo "Component called $name does not exist in $datadir/compounds/" && exit 1
ecf=$datadir/compounds/$name/ecf.xml
cat $ecf | grep -P -o '(?<=audio_filename\=")[^"]*' > $datadir/compounds/$name/files.list
filelist=$datadir/compounds/$name/files.list
[ -f $datadir/compounds/$name/rttm ] && rttm=$datadir/compounds/$name/rttm
[ -f $datadir/compounds/$name/stm ] && stm=$datadir/compounds/$name/stm

if [ -f $ecf ] ; then
  duration=`head -1 $ecf |\
      grep -o -E "duration=\"[0-9]*[    \.]*[0-9]*\"" |\
      perl -e 'while($m=<>) {$m=~s/.*\"([0-9.]+)\".*/\1/; print $m/2.0;}'`
  echo "INFO: Using duration $duration seconds (from ECF)."
else
  echo "WARNING: Using default duration. ECF wasn't specified?"
  duration=9999
fi

inputname=`basename $datadir`
outputname=$name

while (( "$#" )); do
  resultdir=$1;shift
  echo "Processing data directory $resultdir"

  [ ! -d $resultdir ] && echo "Decode dir $resultdir does not exist!" && exit 1;

  targetdir=$resultdir/$outputname

  min_existing=
  max_existing=
  for lmw in `seq $min_lmwt_start $max_lmwt_start`; do
    [ -d $resultdir/score_$lmw ] && [ -z $min_existing ] && min_existing=$lmw
    [ -d $resultdir/score_$lmw ] && [ ! -z $min_existing ] && max_existing=$lmw
  done
  [ -z $min_existing ] && echo "Data directories to be scored could not be found!" && exit 1
  [ -z $max_existing ] && echo "Data directories to be scored could not be found!" && exit 1
  min_lmwt=$min_existing
  max_lmwt=$max_existing
  echo "Found data directories for range LMWT=$min_lmwt:$max_lmwt"

  $cmd LMWT=$min_lmwt:$max_lmwt $targetdir/scoring/filter.LMWT.log \
    set -e';' set -o pipefail';' \
    mkdir -p $targetdir/score_LMWT/';'\
    test -f $resultdir/score_LMWT/$inputname.ctm '&&' \
    utils/filter_scp.pl $filelist $resultdir/score_LMWT/$inputname.ctm '>' \
      $targetdir/score_LMWT/$outputname.ctm || exit 1

  if [ ! -z $stm ] && [ -f $stm ] ; then
    echo "For scoring CTMs, this STM is used $stm"
    local/score_stm.sh --min-lmwt $min_lmwt --max-lmwt $max_lmwt --cer $cer --cmd "$cmd" $datadir/compounds/$name data/lang $targetdir
  else
    echo "Not running scoring, $datadir/compounds/$name/stm does not exist"
  fi


  kws_tasks="kws"

  for kws in `cat $datadir/extra_kws_tasks`; do
    kws_tasks+=" ${kws}_kws"
  done

  for kws in $kws_tasks ; do
    echo "Processing KWS task: $kws"
    mkdir -p $targetdir/$kws
    filter=$targetdir/$kws/utterances
    grep -F -f $filelist $datadir/segments | tee  $targetdir/$kws/segments | \
                       awk '{print $1, $2}' | tee  $targetdir/$kws/utter_map |\
                       awk '{print $1}' > $filter

    kwlist=$datadir/$kws/kwlist.xml

    echo -e  "\tFiltering..."
    #$cmd LMWT=$min_lmwt:$max_lmwt $targetdir/$kws/kws_filter.LMWT.log \
    #  set -e';' set -o pipefail';' \
    #  mkdir -p $targetdir/${kws}_LMWT';'\
    #  cat $resultdir/${kws}_LMWT/'result.*' \| grep -F -f $filter \> $targetdir/${kws}_LMWT/result || exit 1

    $cmd LMWT=$min_lmwt:$max_lmwt $targetdir/$kws/kws_filter.LMWT.log \
      set -e';' set -o pipefail';' \
      mkdir -p $targetdir/${kws}_LMWT';'\
      cat $resultdir/${kws}_LMWT/'result.*' \| utils/filter_scp.pl -f 2 $filter \> $targetdir/${kws}_LMWT/result || exit -1


    echo -e  "\tWrite normalized..."
    $cmd LMWT=$min_lmwt:$max_lmwt $targetdir/$kws/kws_write_normalized.LMWT.log \
      set -e';' set -o pipefail';' \
      cat $targetdir/${kws}_LMWT/result \| \
      utils/write_kwslist.pl --flen=0.01 --duration=$duration \
        --segments=$targetdir/$kws/segments --normalize=true --remove-dup=true\
        --map-utter=$targetdir/$kws/utter_map  --digits=3 - $targetdir/${kws}_LMWT/kwslist.xml || exit 1

    echo -e  "\tWrite unnormalized..."
    $cmd LMWT=$min_lmwt:$max_lmwt $targetdir/$kws/kws_write_unnormalized.LMWT.log \
      set -e';' set -o pipefail';' \
      cat $targetdir/${kws}_LMWT/result \| \
      utils/write_kwslist.pl --flen=0.01 --duration=$duration \
        --segments=$targetdir/$kws/segments --normalize=false --remove-dup=true\
        --map-utter=$targetdir/$kws/utter_map  - $targetdir/${kws}_LMWT/kwslist.unnormalized.xml || exit 1

    if [ ! -z $rttm ] ; then
      echo -e  "\tScoring..."
      $cmd LMWT=$min_lmwt:$max_lmwt $targetdir/$kws/kws_score.LMWT.log \
        set -e';' set -o pipefail';' \
        local/kws_score.sh --ecf $ecf --rttm $rttm --kwlist $kwlist $datadir $targetdir/${kws}_LMWT || exit 1
    else
      echo -e  "\tNot scoring..."
    fi
  done
done
