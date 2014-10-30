#!/bin/bash

. path.sh || exit 1

# use to skip some of steps in this script
stage=1

# number of parallel make jobs
makejobs=4

# Apply patch for gcc 4.7?
# see: https://bugs.debian.org/cgi-bin/bugreport.cgi?bug=667377
apply_gcc_patch=true

. path.sh || exit 1
. utils/parse_options.sh || exit 1

mkdir -p $FEST_ROOT
cp local/lm/est-gcc4.7.patch $FEST_ROOT/
cd $FEST_ROOT

if [ "$stage" -le 1 ]; then
  echo "Downloading source codes and models..."
  wget http://www.cstr.ed.ac.uk/downloads/festival/2.1/speech_tools-2.1-release.tar.gz
  wget http://www.cstr.ed.ac.uk/downloads/festival/2.1/festival-2.1-release.tar.gz
  wget http://www.cstr.ed.ac.uk/downloads/festival/2.1/festlex_CMU.tar.gz
  wget http://www.cstr.ed.ac.uk/downloads/festival/2.1/festlex_OALD.tar.gz
  wget http://www.cstr.ed.ac.uk/downloads/festival/2.1/festlex_POSLEX.tar.gz
  wget http://www.cstr.ed.ac.uk/downloads/festival/2.1/festvox_cmu_us_awb_cg.tar.gz
  wget http://www.cstr.ed.ac.uk/downloads/festival/2.1/festvox_kallpc16k.tar.gz
  wget http://festvox.org/nsw/nsw-0.2.1-current.tar.gz
  wget http://festvox.org/nsw/nsw-data-pc110.tar.bz2
fi


if [ "$stage" -le 2 ]; then
  echo "Untarring the downloaded files..."
  for f in `ls ./*.tar.*`; do
    tar xf $f;
  done
fi


if [ "$stage" -le 3 ]; then
  echo "Compiling the Edinburgh Speech Tools ..."
  cd speech_tools/
  if "$apply_gcc_patch"; then
     info_msg "Applying GCC 4.7 patch..."
     patch -p1 <../est-gcc4.7.patch || exit 1
  fi
  ./configure || exit 1
  make -j $makejobs || exit 1
  cd ..
fi


if [ "$stage" -le 4 ]; then
  echo "Compiling Festival ..."
  cd festival/
  make || exit 1 # Festival seems to have problems with multi-job make
  cd ..
fi


if [ "$stage" -le 5 ]; then
  echo "Compile the NSW package"
  cd nsw/
  cp config/config-dist config/config
  export PATH=$PATH:`pwd`/../festival/bin
  make -j $makejobs || exit 1
  bin/nsw_expand --help
fi

exit 0
