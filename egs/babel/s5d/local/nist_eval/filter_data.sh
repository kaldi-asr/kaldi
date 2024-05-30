. ./path.sh

min_lmwt=5
max_lmwt=25
cer=0
nbest=-1
cmd=run.pl
ntrue_from=
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
  if [ -z $min_existing ] || [ -z $max_existing ] ; then
    for lmw in `seq $min_lmwt_start $max_lmwt_start`; do
      [ -d $resultdir/kwset_kwlist_$lmw ] && [ -z $min_existing ] && min_existing=$lmw
      [ -d $resultdir/kwset_kwlist_$lmw ] && [ ! -z $min_existing ] && max_existing=$lmw
    done
  fi
  [ -z $min_existing ] && echo "Data directories to be scored could not be found!" && exit 1
  [ -z $max_existing ] && echo "Data directories to be scored could not be found!" && exit 1
  min_lmwt=$min_existing
  max_lmwt=$max_existing
  echo "Found data directories for range LMWT=$min_lmwt:$max_lmwt"

  if [ -d $resultdir/score_${min_lmwt} ] ; then
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
  fi


  kws_tasks=""

  for kws in $datadir/kwset_*; do
    kws=`basename $kws`
    echo $kws
    kws_tasks+=" $kws"
  done

  for kws in $kws_tasks ; do
    echo "Processing KWS task: $kws"
    mkdir -p $targetdir/$kws

    echo -e  "\tFiltering... $kws LMWT=$min_lmwt:$max_lmwt"

    indices_dir=$resultdir/kws_indices
    for lmwt in $(seq $min_lmwt $max_lmwt) ; do
      kwsoutput=${targetdir}/${kws}_${lmwt}
      indices=${indices_dir}_$lmwt
      nj=$(cat $indices/num_jobs)

      # This is a memory-efficient way how to do the filtration
      # we do this in this way because the result.* files can be fairly big
      # and we do not want to run into troubles with memory
      files=""
      for job in $(seq 1 $nj); do
        if [ -f $resultdir/${kws}_${lmwt}/result.${job}.gz ] ; then
         files="$files <(gunzip -c $resultdir/${kws}_${lmwt}/result.${job}.gz)"
        elif [ -f $resultdir/${kws}_${lmwt}/result.${job} ] ; then
         files="$files $resultdir/${kws}_${lmwt}/result.${job} "
        else
          echo >&2 "The file $resultdir/${$kws}_${lmwt}/result.${job}[.gz] does not exist"
          exit 1
        fi
      done
      # we have to call it using eval as we need the bash to interpret
      # the (possible) command substitution in case of gz files
      # bash -c would probably work as well, but would spawn another
      # shell instance
      echo $kwsoutput
      echo  $datadir/compounds/$name/utterances
      mkdir -p $kwsoutput
      eval "sort -m -u $files" |\
        int2sym.pl -f 2 $datadir/$kws/utt.map | \
        utils/filter_scp.pl -f 2 $datadir/compounds/$name/utterances |\
        sym2int.pl -f 2 $datadir/$kws/utt.map  |\
        local/search/filter_kws_results.pl --likes --nbest $nbest > $kwsoutput/results || exit 1
    done

    ntrue_from_args=""
    if [ ! -z "$ntrue_from" ]; then
      echo "Using $resultdir/$ntrue_from/$kws for NTRUE"
      ntrue_from_args=" --ntrue-from $resultdir/$ntrue_from/$kws"
    fi
    if [ ! -z $rttm ] ; then
      local/search/score.sh --cmd "$cmd" --extraid ${kws##kwset_}\
        --min-lmwt $min_lmwt --max-lmwt $max_lmwt $ntrue_from_args \
        data/lang $datadir/compounds/$name  ${targetdir}/${kws}  || exit 1;
    elif [ ! -z $ntrue_from ] ; then
      local/search/normalize.sh  --cmd "$cmd" --extraid ${kws##kwset_}\
        --min-lmwt $min_lmwt --max-lmwt $max_lmwt $ntrue_from_args \
        data/lang $datadir/compounds/$name  ${targetdir}/${kws}  || exit 1;
    else
      echo >&2 "Cannot score and don't know which compound set to use to inherit the config"
      exit 1
    fi
  done

done
