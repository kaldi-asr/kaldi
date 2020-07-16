#!/usr/bin/env bash
# Copyright (c) 2015, Johns Hopkins University ( Yenda Trmal <jtrmal@gmail.com> )
# License: Apache 2.0

# Begin configuration section.
# End configuration section
set -o nounset                              # Treat unset variables as an error


if [ ! -x  results ] ; then
  data=$(utils/make_absolute.sh ./local)
  data=$(dirname $data)
  mkdir -p $data/results
  ln -s $data/results results
fi

if [ ! -e ./RESULTS.kws ] ; then
  p=$(basename `utils/make_absolute.sh lang.conf`)
  p=${p##.*}
  filename=kws_results.${p}.${USER}.$(date --iso-8601=seconds)
  echo "#Created on $(date --iso-8601=seconds) by $0" >> results/$filename
  ln -sf  results/$filename RESULTS.kws
fi


set -f
export mydirs=( `find exp/ exp_bnf/ exp_psx/ -name "decode*dev10h.pem*" -type d | sed  's/it[0-9]/*/g;s/epoch[0-9]/*/g' | sort -u` )
set +f
export kwsets=( `find ${mydirs[@]} -type d  -name "kwset*" -not \( -ipath "*syllabs*" -or -path "*phones*" \) | sed 's:.*kwset_::g' | sed 's/_[0-9][0-9]*$//g' | sort -u ` )
(
  #### Word search (converted lattices)
  for kwset in  "${kwsets[@]}"; do
  echo -e "#\n# KWS Task performance (TWV), for the set ["$kwset"] evaluated on $(date --iso-8601=seconds) by user `whoami` on `hostname -f`"
  (
    for f in  "${mydirs[@]}"; do
      find $f -name "metrics.txt" -ipath "*kwset*" -ipath "*_${kwset}_*" -not \( -ipath "*syllabs*" -or -path "*phones*" \)  | xargs grep ATWV | sort -k3,3g | tail -n 1
    done | \
    while IFS='' read -r line || [[ -n "$line" ]]; do
      file=$(echo $line | sed 's/:.*//g' )
      cat $file  | perl -ape 's/ *, */\n/g;' | sed 's/ //g' | grep -E 'TWV|THR'  | paste -s | paste - <(echo $file)
    done
  ) | column -t | sort -k3,3g | \
  (
    while IFS='' read -r line || [[ -n "$line" ]]; do
      echo $line
      f=$(echo $line | rev | awk '{print $1}'| rev)
      d=$(dirname $f)
      echo -ne "\tOOV=0\t"
      local/subset_atwv.pl <(cat data/dev10h.pem/kwset_${kwset}/categories | local/search/filter_by_category.pl data/dev10h.pem/kwset_${kwset}/categories "OOV=0" | cut -f 1 -d ' ' | sort  ) $d/bsum.txt
      echo -ne "\tOOV=1\t"
      local/subset_atwv.pl <(cat data/dev10h.pem/kwset_${kwset}/categories | local/search/filter_by_category.pl data/dev10h.pem/kwset_${kwset}/categories "OOV=1" | cut -f 1 -d ' ' | sort  ) $d/bsum.txt

    done
  )
  done

  #### Syllab search (converted word lattices)
  export kwsets=( `find ${mydirs[@]} -type d  -name "kwset*" -ipath "*syllabs*"  | sed 's:.*kwset_::g' | sed 's/_[0-9][0-9]*$//g' | sort -u ` )
  for kwset in  "${kwsets[@]}"; do
  echo -e "#\n# KWS Task performance (TWV), syllabic search for the set ["$kwset"] evaluated on $(date --iso-8601=seconds) by user `whoami` on `hostname -f`"
  (
    for f in  "${mydirs[@]}"; do
      find $f -name "metrics.txt" -ipath "*kwset*" -ipath "*_${kwset}_*" -ipath "*syllabs*"  | xargs grep ATWV | sort -k3,3g | tail -n 1
    done | \
    while IFS='' read -r line || [[ -n "$line" ]]; do
      file=$(echo $line | sed 's/:.*//g' )
      cat $file  | perl -ape 's/ *, */\n/g;' | sed 's/ //g' | grep -E 'TWV|THR'  | paste -s | paste - <(echo $file)
    done
  ) | column -t | sort -k3,3g | \
  (
    while IFS='' read -r line || [[ -n "$line" ]]; do
      echo $line
      f=$(echo $line | rev | awk '{print $1}'| rev)
      d=$(dirname $f)
      echo -ne "\tOOV=0\t"
      local/subset_atwv.pl <(cat data/dev10h.pem/kwset_${kwset}/categories | local/search/filter_by_category.pl data/dev10h.pem/kwset_${kwset}/categories "OOV=0" | cut -f 1 -d ' ' | sort  ) $d/bsum.txt
      echo -ne "\tOOV=1\t"
      local/subset_atwv.pl <(cat data/dev10h.pem/kwset_${kwset}/categories | local/search/filter_by_category.pl data/dev10h.pem/kwset_${kwset}/categories "OOV=1" | cut -f 1 -d ' ' | sort  ) $d/bsum.txt

    done
  )
  done


  #### Phone search (converted word lattices)
  export kwsets=( `find ${mydirs[@]} -type d  -name "kwset*" -ipath "*phones*"  | sed 's:.*kwset_::g' | sed 's/_[0-9][0-9]*$//g' | sort -u ` )
  for kwset in  "${kwsets[@]}"; do
  echo -e "#\n# KWS Task performance (TWV), phonetic search for the set ["$kwset"] evaluated on $(date --iso-8601=seconds) by user `whoami` on `hostname -f`"
  (
    for f in  "${mydirs[@]}"; do
      find $f -name "metrics.txt" -ipath "*kwset*" -ipath "*_${kwset}_*" -ipath "*phones*" | xargs grep ATWV | sort -k3,3g | tail -n 1
    done | \
    while IFS='' read -r line || [[ -n "$line" ]]; do
      file=$(echo $line | sed 's/:.*//g' )
      cat $file  | perl -ape 's/ *, */\n/g;' | sed 's/ //g' | grep -E 'TWV|THR'  | paste -s | paste - <(echo $file)
    done
  ) | column -t | sort -k3,3g | \
  (
    while IFS='' read -r line || [[ -n "$line" ]]; do
      echo $line
      f=$(echo $line | rev | awk '{print $1}'| rev)
      d=$(dirname $f)
      echo -ne "\tOOV=0\t"
      local/subset_atwv.pl <(cat data/dev10h.pem/kwset_${kwset}/categories | local/search/filter_by_category.pl data/dev10h.pem/kwset_${kwset}/categories "OOV=0" | cut -f 1 -d ' ' | sort  ) $d/bsum.txt
      echo -ne "\tOOV=1\t"
      local/subset_atwv.pl <(cat data/dev10h.pem/kwset_${kwset}/categories | local/search/filter_by_category.pl data/dev10h.pem/kwset_${kwset}/categories "OOV=1" | cut -f 1 -d ' ' | sort  ) $d/bsum.txt

    done
  )

  done


  set -f
  export mydirs=( `find exp/ exp_bnf/ exp_psx/ -name "decode*dev10h.syll.pem*" -type d | sed  's/it[0-9]/*/g;s/epoch[0-9]/*/g' | sort -u` )
  set +f
  if [ ! -z ${mydirs+x} ] ; then
  export kwsets=( `find ${mydirs[@]} -type d  -name "kwset*" -not \( -ipath "*syllabs*" -or -path "*phones*" \) | sed 's:.*kwset_::g' | sed 's/_[0-9][0-9]*$//g' | sort -u ` )
  #declare -p kwsets
  for kwset in  "${kwsets[@]}"; do
  echo -e "#\n# KWS Task performance (TWV), syllabic decode+search for the set ["$kwset"] evaluated on $(date --iso-8601=seconds) by user `whoami` on `hostname -f`"
  (
    for f in  "${mydirs[@]}"; do
      find $f -name "metrics.txt" -ipath "*kwset*" -ipath "*_${kwset}_*" -not \( -ipath "*syllabs*" -or -path "*phones*" \)  | xargs grep ATWV | sort -k3,3g | tail -n 1
    done | \
    while IFS='' read -r line || [[ -n "$line" ]]; do
      file=$(echo $line | sed 's/:.*//g' )
      cat $file  | perl -ape 's/ *, */\n/g;' | sed 's/ //g' | grep -E 'TWV|THR'  | paste -s | paste - <(echo $file)
    done
  ) | column -t | sort -k3,3g | \
  (
    while IFS='' read -r line || [[ -n "$line" ]]; do
      echo $line
      f=$(echo $line | rev | awk '{print $1}'| rev)
      d=$(dirname $f)
      echo -ne "\tOOV=0\t"
      local/subset_atwv.pl <(cat data/dev10h.pem/kwset_${kwset}/categories | local/search/filter_by_category.pl data/dev10h.pem/kwset_${kwset}/categories "OOV=0" | cut -f 1 -d ' ' | sort  ) $d/bsum.txt
      echo -ne "\tOOV=1\t"
      local/subset_atwv.pl <(cat data/dev10h.pem/kwset_${kwset}/categories | local/search/filter_by_category.pl data/dev10h.pem/kwset_${kwset}/categories "OOV=1" | cut -f 1 -d ' ' | sort  ) $d/bsum.txt

    done
  )

  done
  fi

  set -f
  export mydirs=( `find exp/ exp_bnf/ exp_psx/ -name "decode*dev10h.phn.pem*" -type d | sed  's/it[0-9]/*/g;s/epoch[0-9]/*/g' | sort -u` )
  set +f
  if [ ! -z ${mydirs+x} ] ; then
  export kwsets=( `find ${mydirs[@]} -type d  -name "kwset*" -not \( -ipath "*syllabs*" -or -path "*phones*" \) | sed 's:.*kwset_::g' | sed 's/_[0-9][0-9]*$//g' | sort -u ` )
  #declare -p kwsets
  for kwset in  "${kwsets[@]}"; do
  echo -e "#\n# KWS Task performance (TWV), phonetic decode+search for the set ["$kwset"] evaluated on $(date --iso-8601=seconds) by user `whoami` on `hostname -f`"
  (
    for f in  "${mydirs[@]}"; do
      find $f -name "metrics.txt" -ipath "*kwset*" -ipath "*_${kwset}_*" -not \( -ipath "*syllabs*" -or -path "*phones*" \)  | xargs grep ATWV | sort -k3,3g | tail -n 1
    done | \
    while IFS='' read -r line || [[ -n "$line" ]]; do
      file=$(echo $line | sed 's/:.*//g' )
      cat $file  | perl -ape 's/ *, */\n/g;' | sed 's/ //g' | grep -E 'TWV|THR'  | paste -s | paste - <(echo $file)
    done
  ) | column -t | sort -k3,3g | \
  (
    while IFS='' read -r line || [[ -n "$line" ]]; do
      echo $line
      f=$(echo $line | rev | awk '{print $1}'| rev)
      d=$(dirname $f)
      echo -ne "\tOOV=0\t"
      local/subset_atwv.pl <(cat data/dev10h.pem/kwset_${kwset}/categories | local/search/filter_by_category.pl data/dev10h.pem/kwset_${kwset}/categories "OOV=0" | cut -f 1 -d ' ' | sort  ) $d/bsum.txt
      echo -ne "\tOOV=1\t"
      local/subset_atwv.pl <(cat data/dev10h.pem/kwset_${kwset}/categories | local/search/filter_by_category.pl data/dev10h.pem/kwset_${kwset}/categories "OOV=1" | cut -f 1 -d ' ' | sort  ) $d/bsum.txt

    done
  )

  done
  fi
) | tee  RESULTS.kws
