#! /bin/bash

set -e 
set -o pipefail
set -u
set -x

if [ $# -ne 2 ]; then
  echo "Usage: $0 <filelist> <dir>"
  exit 1
fi

filelist=$1
dir=$2

export PATH=local/data_prep/csr_hub4_utils:$PATH

for file in `cat $filelist`; do
	BASENM=`basename $file`
  name="${BASENM%.*}"

	echo "Running LM pipeline for |$BASENM|..." 1>&2
  gunzip -c $file | pare-sgml.perl | \
    bugproc.perl | \
    numhack.perl | \
    numproc.perl -xlocal/data_prep/csr_hub4_utils/num_excp | \
    abbrproc.perl local/data_prep/csr_hub4_utils/abbrlist | \
    puncproc.perl -np | gzip -c > $dir/$name.txt.gz
	echo "Done with $BASENM."
done
