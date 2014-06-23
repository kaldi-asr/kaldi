#!/bin/bash

# Copyright, University of Edinburgh (Pawel Swietojanski and Jonathan Kilgour)

if [ $# -ne 1 ]; then
  echo "Usage: $0 <ami-dir>"
  exit 1;
fi

adir=$1
wdir=data/local/annotations

[ ! -f $adir/annotations/AMI-metadata.xml ] && echo "$0: File $adir/annotations/AMI-metadata.xml no found." && exit 1;

mkdir -p $wdir/log

if [ ! -d $wdir/nxt ]; then
  echo "Downloading NXT annotation tool..."
  wget -O $wdir/nxt.zip http://sourceforge.net/projects/nite/files/nite/nxt_1.4.4/nxt_1.4.4.zip &> /dev/null
  unzip -d $wdir/nxt $wdir/nxt.zip &> /dev/null
fi

if [ ! -f $wdir/transcripts0 ]; then
  echo "Parsing XML files (can take several minutes)..."
  nxtlib=$wdir/nxt/lib
  java -cp $nxtlib/nxt.jar:$nxtlib/xmlParserAPIs.jar:$nxtlib/xalan.jar:$nxtlib \
     FunctionQuery -c $adir/annotations/AMI-metadata.xml -q '($s segment)(exists $w1 w):$s^$w1' -atts obs who \
     '@extract(($sp speaker)($m meeting):$m@observation=$s@obs && $m^$sp & $s@who==$sp@nxt_agent,global_name, 0)'\
     '@extract(($sp speaker)($m meeting):$m@observation=$s@obs && $m^$sp & $s@who==$sp@nxt_agent, channel, 0)' \
     transcriber_start transcriber_end starttime endtime '$s' '@extract(($w w):$s^$w & $w@punc="true", starttime,0,0)' \
     1> $wdir/transcripts0 2> $wdir/log/nxt_export.log
fi

#remove NXT logs dumped to stdio
grep -e '^Found' -e '^Obs' -i -v $wdir/transcripts0 > $wdir/transcripts1

exit 0;

