#!/bin/bash

# Copyright 2013 MERL (author: Shinji Watanabe)
# Contains some code by Microsoft Corporation, Johns Hopkins University (author: Daniel Povey)

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.

if [ $# -ne 2 ]; then
  printf "\nUSAGE: %s <wsjcam0 data dir> <dest dir>\n\n" `basename $0`
  echo "e.g.,:"
  echo " `basename $0` /archive/speech-db/processed/public/REVERB/wsjcam0 data_mc_tr"
  exit 1;
fi

wsjcam0_dir=$1
reverb_tr_dir=$2

dir=`pwd`/data/local/reverb_tools
mkdir -p $dir $reverb_tr_dir
lmdir=`pwd`/data/local/nist_lm

# Download tools
URL1="http://reverb2014.dereverberation.com/tools/reverb_tools_for_Generate_mcTrainData.tgz"
URL2="http://reverb2014.dereverberation.com/tools/REVERB_TOOLS_FOR_ASR_ver2.0.tgz"
for f in $URL1 $URL2; do
  x=`basename $f`
  if [ ! -e $dir/$x ]; then
    wget $f -O $dir/$x || exit 1;
    tar zxvf $dir/$x -C $dir || exit 1;
  fi
done
URL3="http://reverb2014.dereverberation.com/tools/taskFiles_et.tgz"
x=`basename $URL3`
if [ ! -e $dir/$x ]; then
  wget $URL3 -O $dir/$x || exit 1;
  tar zxvf $dir/$x -C $dir || exit 1;
  cp -fr $dir/`basename $x .tgz`/* $dir/ReleasePackage/reverb_tools_for_asr_ver2.0/taskFiles/
fi

# Download and install nist tools
pushd $dir/ReleasePackage/reverb_tools_for_asr_ver2.0
sed -e "s|^main$|targetSPHEREDir\=tools/SPHERE\ninstall_nist|" installTools > installnist
chmod u+x installnist
./installnist
popd

# Make mcTrainData
cp local/Generate_mcTrainData_cut.m $dir/reverb_tools_for_Generate_mcTrainData/
pushd $dir/reverb_tools_for_Generate_mcTrainData/
# copied nist tools required for the following matlab command
cp $dir/ReleasePackage/reverb_tools_for_asr_ver2.0/tools/SPHERE/nist/bin/{h_strip,w_decode} ./bin/

tmpdir=`mktemp -d tempXXXXX `
tmpmfile=$tmpdir/run_mat.m
cat <<EOF > $tmpmfile
addpath(genpath('.'))
Generate_mcTrainData_cut('$wsjcam0_dir', '$reverb_tr_dir');
EOF
cat $tmpmfile | matlab -nodisplay
rm -rf $tmpdir
popd

echo "Successfully generated multi-condition training data and stored it in $reverb_tr_dir." && exit 0;
