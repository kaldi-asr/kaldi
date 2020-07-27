#!/usr/bin/env bash
#
# Copyright  2018  Johns Hopkins University (Author: Shinji Watanabe)
# Apache 2.0
# This script is adapted from data preprations scripts in the Kaldi reverb recipe
# https://github.com/kaldi-asr/kaldi/tree/master/egs/reverb/s5/local

# Begin configuration section.
wavdir=${PWD}/wav
# End configuration section

. ./utils/parse_options.sh  # accept options.. you can run this run.sh with the

. ./path.sh

echo >&2 "$0" "$@"
if [ $# -ne 1 ] ; then
  echo >&2 "$0" "$@"
  echo >&2 "$0: Error: wrong number of arguments"
  echo -e >&2 "Usage:\n  $0 [opts] <wsjcam0-dir>"
  echo -e >&2 "eg:\n  $0 /export/corpora3/LDC/LDC95S24/wsjcam0"
  exit 1
fi

set -e -o pipefail

wsjcam0=$1
mkdir -p ${wavdir}

# tool directory
dir=${PWD}/data/local/reverb_tools
mkdir -p ${dir}

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

# generate WAV files for matlab
echo "generating WAV files"
sph2pipe=$KALDI_ROOT/tools/sph2pipe_v2.5/sph2pipe
if [ ! -x $sph2pipe ]; then
    echo "Could not find (or execute) the sph2pipe program at ${sph2pipe}";
    exit 1;
fi
for sph in `cat ${dir}/reverb_tools_for_Generate_mcTrainData/etc/audio_si_tr.lst`; do
    d=`dirname ${wavdir}/WSJCAM0/data/${sph}`
    if [ ! -d "${d}" ]; then
	mkdir -p ${d}
    fi
    ${sph2pipe} -f wav ${wsjcam0}/data/${sph}.wv1 > ${wavdir}/WSJCAM0/data/${sph}.wav
done
nwav=`find ${wavdir}/WSJCAM0/data/primary_microphone/si_tr | grep .wav | wc -l`
echo "generated ${nwav} WAV files (it must be 7861)"
[ "$nwav" -eq 7861 ] || echo "Warning: expected 7861 WAV files, got $nwav"

# generalte training data
reverb_tr_dir=${wavdir}/REVERB_WSJCAM0_tr
cp local/Generate_mcTrainData_cut.m $dir/reverb_tools_for_Generate_mcTrainData/
pushd $dir/reverb_tools_for_Generate_mcTrainData/
tmpdir=`mktemp -d tempXXXXX `
tmpmfile=$tmpdir/run_mat.m
cat <<EOF > $tmpmfile
addpath(genpath('.'))
Generate_mcTrainData_cut('$wavdir/WSJCAM0', '$reverb_tr_dir');
EOF
cat $tmpmfile | matlab -nodisplay
rm -rf $tmpdir
popd

echo "Successfully generated multi-condition training data and stored it in $reverb_tr_dir." && exit 0;
