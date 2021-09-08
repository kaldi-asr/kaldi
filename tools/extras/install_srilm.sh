#!/usr/bin/env bash

current_path=`pwd`
current_dir=`basename "$current_path"`

if [ "tools" != "$current_dir" ]; then
    echo "You should run this script in tools/ directory!!"
    exit 1
fi

if [ ! -d liblbfgs-1.10 ]; then
    echo Installing libLBFGS library to support MaxEnt LMs
    bash extras/install_liblbfgs.sh || exit 1
fi

! command -v gawk > /dev/null && \
   echo "GNU awk is not installed so SRILM will probably not work correctly: refusing to install" && exit 1;

if [ $# -ne 3 ]; then
    echo "SRILM download requires some information about you"
    echo
    echo "Usage: $0 <name> <organization> <email>"
    exit 1
fi

srilm_url="http://www.speech.sri.com/projects/srilm/srilm_download.php"
post_data="WWW_file=srilm-1.7.3.tar.gz&WWW_name=$1&WWW_org=$2&WWW_email=$3"

if ! wget --post-data "$post_data" -O ./srilm.tar.gz "$srilm_url"; then
    echo 'There was a problem downloading the file.'
    echo 'Check you internet connection and try again.'
    exit 1
fi

mkdir -p srilm
cd srilm


if [ -f ../srilm.tgz ]; then
    tar -xvzf ../srilm.tgz # Old SRILM format
elif [  -f ../srilm.tar.gz ]; then
    tar -xvzf ../srilm.tar.gz # Changed format type from tgz to tar.gz
fi

major=`gawk -F. '{ print $1 }' RELEASE`
minor=`gawk -F. '{ print $2 }' RELEASE`
micro=`gawk -F. '{ print $3 }' RELEASE`

if [ $major -le 1 ] && [ $minor -le 7 ] && [ $micro -le 1 ]; then
  echo "Detected version 1.7.1 or earlier. Applying patch."
  patch -p0 < ../extras/srilm.patch
fi

# set the SRILM variable in the top-level Makefile to this directory.
cp Makefile tmpf

cat tmpf | gawk -v pwd=`pwd` '/SRILM =/{printf("SRILM = %s\n", pwd); next;} {print;}' \
  > Makefile || exit 1
rm tmpf

mtype=`sbin/machine-type`

echo HAVE_LIBLBFGS=1 >> common/Makefile.machine.$mtype
grep ADDITIONAL_INCLUDES common/Makefile.machine.$mtype | \
    sed 's|$| -I$(SRILM)/../liblbfgs-1.10/include|' \
    >> common/Makefile.machine.$mtype

grep ADDITIONAL_LDFLAGS common/Makefile.machine.$mtype | \
    sed 's|$| -L$(SRILM)/../liblbfgs-1.10/lib/ -Wl,-rpath -Wl,$(SRILM)/../liblbfgs-1.10/lib/|' \
    >> common/Makefile.machine.$mtype

make || exit

cd ..
(
  [ ! -z "${SRILM}" ] && \
    echo >&2 "SRILM variable is aleady defined. Undefining..." && \
    unset SRILM

  [ -f ./env.sh ] && . ./env.sh

  [ ! -z "${SRILM}" ] && \
    echo >&2 "SRILM config is already in env.sh" && exit

  wd=`pwd`
  wd=`readlink -f $wd || pwd`

  echo "export SRILM=$wd/srilm"
  dirs="\${PATH}"
  for directory in $(cd srilm && find bin -type d ) ; do
    dirs="$dirs:\${SRILM}/$directory"
  done
  echo "export PATH=$dirs"
) >> env.sh

echo >&2 "Installation of SRILM finished successfully"
echo >&2 "Please source the tools/env.sh in your path.sh to enable it"
