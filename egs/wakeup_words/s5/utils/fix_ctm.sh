#! /bin/bash

stmfile=$1
ctmfile=$2

segments_stm=`cat $stmfile | cut -f 1 -d ' ' | sort -u`
segments_ctm=`cat $ctmfile | cut -f 1 -d ' ' | sort -u`

segments_stm_count=`echo "$segments_stm" | wc -l `
segments_ctm_count=`echo "$segments_ctm" | wc -l `

#echo $segments_stm_count
#echo $segments_ctm_count

if [ "$segments_stm_count" -gt "$segments_ctm_count"  ] ; then
  pp=$( diff <(echo "$segments_stm") <(echo "$segments_ctm" ) | grep "^<" | sed "s/^< *//g")
  (
    for elem in $pp ; do
      echo "$elem 1 0 0 EMPTY_RECOGNIZED_PHRASE"
    done
  ) >> $ctmfile
  echo "FIXED CTM FILE"
  exit 0
elif [ "$segments_stm_count" -lt "$segments_ctm_count"  ] ; then
  echo "Segment STM count: $segments_stm_count"
  echo "Segment CTM count: $segments_ctm_count"
  echo "FAILURE FIXING CTM FILE"
  exit 1
else
  exit 0
fi

