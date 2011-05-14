#!/bin/bash

INFILE=$1
CMDLINE="$@"

cat $INFILE | awk -v cmdargs="$CMDLINE" \
'BEGIN{
  fn_att="reconet.txt"
  delete stateLabel
  delete wordLabel
  delete link
  delete score
  delete fstate
}

# write link file
$1~/^[0-9]/{
  for (i=3;i<=NF;i++){
    if (!($i~/=/)) {link[$1"\t"$i]=$1;dest=i}
    else {
      sub(/^l=/,"",$i)
      sub(/-/,"",$i)
      if ($1"\t"$dest in score) {
        score[$1"\t"$dest]=score[$1"\t"$dest]","$i
      } else {
        score[$1"\t"$dest]=$i
      }
    }
  }
}

$1~/^[0-9]/{
  #save state label output

  if ($2~/^M=/) {
    # phone model labels
    stateLabel[$1]=$2;
    sub(/^M=/,"",stateLabel[$1]);
    # strip left/right context of triphones
    #sub(/^[a-z]+-/,"",stateLabel[$1])
    #sub(/\+[a-z]+$/,"",stateLabel[$1])
  } else {
    stateLabel[$1]="eps";
  }

  if ($2~/^W=/) {
    # word labels
    wordLabel[$1]=$2;
    sub(/W=!NULL/,"eps",wordLabel[$1])     # replace by eps
    sub(/^W=/,"",wordLabel[$1])
    gsub(/"/,"",wordLabel[$1])
  } else {
    wordLabel[$1]="eps";
  }

  # save final state(s)
  if (NF==2) {fstate[$1]==$1}
}

# end of record
END{
  #write the WFST link file
  for (l in link) { 
    print l "\t" stateLabel[link[l]] "\t" wordLabel[link[l]] "\t" score[l]| " sort -n -k 1,3 > "fn_att 
  }
  for (fs in fstate) { print fs | " sort -n -k 1,3 > "fn_att }
  close(" sort -n -k 1,3 > "fn_att);
  }
'

cat reconet.txt | awk '($4!="") {if ($4!="eps") {print $4}}' | sort | uniq | awk 'BEGIN{cnt=1; print "eps","0"} {print $1,cnt; cnt=cnt+1}' >reconet.words