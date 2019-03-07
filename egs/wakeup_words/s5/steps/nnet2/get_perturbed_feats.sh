#!/bin/bash


# begin configuration section

cmd="run.pl"
num_copies=5  # support 3, 4 or 5 perturbed copies of the data.
stage=0
nj=8
cleanup=true
feature_type=fbank
# end configuration section

set -e
. utils/parse_options.sh 

if [ $# -ne 5 ]; then
  echo "Usage: $0 [options] <baseline-feature-config> <feature-storage-dir> <log-location> <input-data-dir> <output-data-dir> "
  echo "e.g.: $0 conf/fbank_40.conf mfcc exp/perturbed_fbank_train data/train data/train_perturbed_fbank"
  echo "Supported options: "
  echo "--feature-type (fbank|mfcc|plp)  # Type of features we are making, default fbank"
  echo "--cmd 'command-program'      # Mechanism to run jobs, e.g. run.pl"
  echo "--num-copies <n>             # Number of perturbed copies of the data (support 3, 4 or 5), default 5"
  echo "--stage <stage>              # Use for partial re-run"
  echo "--cleanup (true|false)       # If false, do not clean up temp files (default: true)"
  echo "--nj <num-jobs>              # How many jobs to use for feature extraction (default: 8)"
  exit 1;
fi

base_config=$1
featdir=$2
dir=$3 # dir/log* will contain log-files
inputdata=$4
data=$5

# Set pairs of (VTLN warp factor, time-warp factor)
# Aim to put these roughly in a circle centered at 1.0-1.0; the
# dynamic range of the VTLN warp factor will be 0.9 to 1.1 and
# of the time-warping factor will be 0.8 to 1.2.
if [ $num_copies -eq 5 ]; then
  pairs="1.1-1.0 1.05-1.2 1.0-0.8 0.95-1.1 0.9-0.9" 
elif [ $num_copies -eq 4 ]; then
  pairs="1.1-1.0 1.0-0.8 1.0-1.2 0.9-1.0"
elif [ $num_copies -eq 3 ]; then
  pairs="1.1-1.1 1.0-0.8 0.9-1.1"
else
  echo "$0: unsupported --num-copies value: $num_copies (support 3, 4 or 5)"
fi

for f in $base_config $inputdata/wav.scp; do 
  if [ ! -f $f ]; then
    echo "Expected file $f to exist"
    exit 1;
  fi
done

if [ "$feature_type" != "fbank" ] && [ "$feature_type" != "mfcc" ] && \
   [ "$feature_type" != "plp" ]; then 
  echo "$0: Invalid option --feature-type=$feature_type"
  exit 1;
fi

mkdir -p $featdir
mkdir -p $dir/conf $dir/log

all_feature_dirs=""

for pair in $pairs; do
  vtln_warp=`echo $pair | cut -d- -f1`
  time_warp=`echo $pair | cut -d- -f2`
  fs=`perl -e "print ($time_warp*10);"`
  conf=$dir/conf/$pair.conf
  this_dir=$dir/$pair
  
  ( cat $base_config; echo; echo "--frame-shift=$fs"; echo "--vtln-warp=$vtln_warp" ) > $conf
  
  echo "Making ${feature_type} features for VTLN-warp $vtln_warp and time-warp $time_warp"

  feature_data=${data}-$pair
  all_feature_dirs="$all_feature_dirs $feature_data"

  utils/copy_data_dir.sh --spk-prefix ${pair}- --utt-prefix ${pair}- $inputdata $feature_data
  steps/make_${feature_type}.sh --${feature_type}-config $conf --nj "$nj" --cmd "$cmd" $feature_data $this_dir $featdir

  steps/compute_cmvn_stats.sh $feature_data $this_dir $featdir
done

utils/combine_data.sh $data $all_feature_dirs


# In the combined feature directory, create a file utt2uniq which maps
# our extended utterance-ids to "unique utterances".  This enables the
# script steps/nnet2/get_egs.sh to hold out data in a more proper way.
cat $data/utt2spk | \
   perl -e ' while(<STDIN>){ @A=split; $x=shift @A; $y=$x; 
     foreach $pair (@ARGV) { $y =~ s/^${pair}-// && last; } print "$x $y\n"; } ' $pairs \
  > $data/utt2uniq

if $cleanup; then
  echo "$0: Cleaning up temporary directories for ${feature_type} features."
  # Note, this just removes the .scp files and so on, not the data which is located in
  # $featdir and which is still needed.
  rm -r $all_feature_dirs
fi
