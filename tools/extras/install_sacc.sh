#!/usr/bin/env bash

WGET=${WGET:-wget}

# Make sure we are in the tools/ directory.
if [ `basename $PWD` == extras ]; then
  cd ..
fi

! [ `basename $PWD` == tools ] && \
   echo "You must call this script from the tools/ directory" && exit 1;

mkdir -p pitch_trackers/sacc

cd pitch_trackers/sacc
if [ -s SAcC_GLNXA64.zip ]; then
  echo "*SAcC_GLNXA64.zip already exists, not getting it."
elif [ -d "$DOWNLOAD_DIR" ]; then
  cp -p "$DOWNLOAD_DIR/SAcC_GLNXA64.zip" . || exit 1
else
  ! $WGET -t 2 https://labrosa.ee.columbia.edu/projects/SAcC/SAcC_GLNXA64.zip && \
    echo "Error wgetting SAcC_GLNXA64.zip" && exit 1;
fi

if [ -d SAcC_GLNXA64 ]; then
  echo "*It looks like SAcC_GLNXA64.zip has already been unzipped, not unzipping it."
else
  ! unzip SAcC_GLNXA64.zip && echo "Error unzipping SAcC_GLNXA64.zip [e.g. unzip not installed?]" \
   && exit 1;
fi

if [ -f MCRInstaller_glnxa64.bin ]; then
  echo "*It looks like you already downloaded MCRInstaller_glnxa64.bin, not getting it."
elif [ -d "$DOWNLOAD_DIR" ]; then
  cp -p "$DOWNLOAD_DIR/MCRInstaller_glnxa64.bin" . || exit 1
else
  ! $WGET -t 2 https://www.ee.columbia.edu/~dpwe/tmp/MCRInstaller_glnxa64.bin && \
   echo "Error getting MCRInstaller_glnxa64.bin" && exit 1;
fi

chmod +x ./MCRInstaller_glnxa64.bin

if [ ! -d matlab_runtime ]; then
  echo "*Running the Matlab runtime installer"
  echo "*It can take some time to finish (~10 minutes), be patient"
  echo "*Command: ./MCRInstaller_glnxa64.bin -silent -P installLocation=\"`pwd`/matlab_runtime\""
  ./MCRInstaller_glnxa64.bin -silent -P installLocation="`pwd`/matlab_runtime"
else
  echo "*It looks like the Matlab runtime has already been installed, not installing it."
fi

if [ ! -d matlab_runtime ]; then
  echo "*Error: the directory matlab_runtime does not exist, something went wrong in the"
  echo "*Matlab runtime installer."
  exit 1;
fi

if [ ! -f SAcC_GLNXA64/run_SAcC.sh ]; then
  echo "Something went wrong: SAcC_GLNXA64/run_SAcC.sh does not exist."
  exit 1;
fi

if ! grep matlab_runtime SAcC_GLNXA64/run_SAcC.sh >/dev/null; then
  echo "Replacing the MCRROOT variable in SAcC_GLNXA64/run_SAcC.sh"
  cp SAcC_GLNXA64/run_SAcC.sh SAcC_GLNXA64/run_SAcC.sh.bak || exit 1;
  ! cat SAcC_GLNXA64/run_SAcC.sh.bak | \
    awk -v pwd=`pwd` '/^MCRROOT=/{ printf("MCRROOT=%s/matlab_runtime/v714\n", pwd);  next; } {print;}' \
    > SAcC_GLNXA64/run_SAcC.sh && echo "Error replacing the MCRROOT variable in script, restoring from archive" \
    && unzip -o SAcC_GLNXA64.zip SAcC_GLNXA64/run_SAcC.sh && exit 1;
else
  echo "*Not replacing MCRROOT variable in script, it was already done."
fi

echo "Testing SAcC."
cd SAcC_GLNXA64
! ./run_SAcC.sh files.list conf/Babelnet_sr8k_bpo6_sb24_k10.config \
  && echo "**Error testing SAcC-- something went wrong." && exit 1;

echo "Test succeeded."
exit 0;
