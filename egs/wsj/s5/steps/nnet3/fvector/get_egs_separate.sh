#!/bin/bash

# This script dumps training examples (egs) for fvector training. At least,
# each eg has two "NnetIo"s(data-chunks), which come from the same original 
# source signal fragment. The two data-chunks in each eg will have respectively
# n=0 and n=1.
#
#
# This script, which will generally be called from other neural-net training
# scripts, extracts the training examples used to train the neural net (and also
# the validation examples used for diagnostics), and puts them in archives.

# Begin configuration section.
cmd=run.pl
egs_per_iter=12500     # have this many frames per archive.
egs_per_iter_diagnostic=10000    # have this many frames per achive for the
                                 # archives used for diagnostics.
num_diagnostic_percent=5   # we want to test the training and validation likelihoods
                           # on a range of utterance lengths, and this number
                           # controls how many archives we evaluate on. Select
                           # "num_diagnostic_percent"% train data to be valid
chunk_size=120
compress=true
srand=0
generate_egs_scp=true

stage=0
nj=8    # This should be set to the maximum number of jobs you are confortable
        # to run in parallel

echo "$0 $@"

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 3 ]; then
  echo "Usage: $0 [opts] <data-dir> <noise-dir> <egs-dir>"
  echo " e.g.: $0 data/train data/noise exp/fvector/egs"
  echo ""
  echo "Main options (for others, see top of script file)"
  echo "  --nj <nj>                                        # The maximum number of jobs you want to run in"
  echo "                                                   # parallel (increase this only if you have good disk and"
  echo "                                                   # network speed).  default=8"
  echo "  --cmd (utils/run.pl;utils/queue.pl <queue opts>) # how to run jobs."  
  echo "  --frames-per-iter <#samples;100000>              # Target number of frames per archive"
  echo "                                                   # {train_subset,valid}.*.egs"
  echo "  --stage <stage|0>                                # Used to run a partially-completed training process from"
  echo "                                                   # somewhere in the middle."
  echo ""

  exit 1;
fi

data_dir=$1
noise_dir=$2
egs_dir=$3

for f in $data_dir/wav.scp $noise_dir/wav.scp $noise_dir/utt2dur_fix; do
  [ ! -f $f ] && echo "$0: expected file $f to exist" && exit 1;
done

mkdir -p $egs_dir
mkdir -p $egs_dir/log
mkdir -p $egs_dir/info
num_utts=$(cat $data_dir/wav.scp | wc -l)
num_valid=$[$num_utts*$num_diagnostic_percent/100];

#Assume recording-id == utt-id
if [ $stage -le 1 ]; then
  #Get list of validation utterances.
  awk '{print $1}' $data_dir/wav.scp | utils/shuffle_list.pl | head -$num_valid \
    > ${egs_dir}/info/valid_uttlist
  cat $data_dir/wav.scp | utils/filter_scp.pl --exclude $egs_dir/info/valid_uttlist | \
    awk '{print $1}' > ${egs_dir}/info/train_uttlist
  cat ${egs_dir}/info/train_uttlist | utils/shuffle_list.pl | head -$num_valid \
    > ${egs_dir}/info/train_diagnostic_uttlist
fi
# get the (120ms) chunks from wav.scp and noise.scp. And compose 1 source
# chunk and 2 noise chunks into a matrix.
if [ $stage -le 2 ]; then
  sdata=$data_dir/split$nj
  utils/data/split_data.sh $data_dir $nj || exit 1;
  $cmd JOB=1:$nj $egs_dir/log/cut_train_wav_into_chunks.JOB.log \
    fvector-chunk-separate --chunk-size=$chunk_size "scp:utils/filter_scp.pl --exclude $egs_dir/info/valid_uttlist $sdata/JOB/wav.scp |" \
      scp:$noise_dir/wav.scp $noise_dir/utt2dur_fix \
      ark,scp:$egs_dir/orign_train_chunks.JOB.ark,$egs_dir/orign_train_chunks.JOB.scp \
      ark,scp:$egs_dir/orign_train_noise_chunks.JOB.ark,$egs_dir/orign_train_noise_chunks.JOB.scp

  for n in $(seq $nj); do
    cat $egs_dir/orign_train_chunks.${n}.scp || exit 1;
  done > $data_dir/orign_train_chunks.all.scp
  for n in $(seq $nj); do
    cat $egs_dir/orign_train_noise_chunks.${n}.scp || exit 1;
  done > $data_dir/orign_train_noise_chunks.all.scp
  cp $data_dir/orign_train_chunks.all.scp $egs_dir/orign_train_chunks.all.scp
  cp $data_dir/orign_train_noise_chunks.all.scp $egs_dir/orign_train_noise_chunks.all.scp

  $cmd $egs_dir/log/cut_valid_wav_into_chunks.log \
    fvector-chunk-separate --chunk-size=$chunk_size "scp:utils/filter_scp.pl $egs_dir/info/valid_uttlist $data_dir/wav.scp |" \
      scp:$noise_dir/wav.scp $noise_dir/utt2dur_fix \
      ark,scp:$egs_dir/orign_valid_chunks.ark,$egs_dir/orign_valid_chunks.scp \
      ark,scp:$egs_dir/orign_valid_noise_chunks.ark,$egs_dir/orign_valid_noise_chunks.scp
  cp $egs_dir/orign_valid_chunks.scp $data_dir/orign_valid_chunks.scp
  cp $egs_dir/orign_valid_noise_chunks.scp $data_dir/orign_valid_noise_chunks.scp

  $cmd $egs_dir/log/cut_train_diagnostic_wav_into_chunks.log \
    fvector-chunk-separate --chunk-size=$chunk_size "scp:utils/filter_scp.pl $egs_dir/info/train_diagnostic_uttlist $data_dir/wav.scp |" \
      scp:$noise_dir/wav.scp $noise_dir/utt2dur_fix \
      ark,scp:$egs_dir/orign_train_diagnostic_chunks.ark,$egs_dir/orign_train_diagnostic_chunks.scp \
      ark,scp:$egs_dir/orign_train_diagnostic_noise_chunks.ark,$egs_dir/orign_train_diagnostic_noise_chunks.scp
  cp $egs_dir/orign_train_diagnostic_chunks.scp $data_dir/orign_train_diagnostic_chunks.scp
  cp $egs_dir/orign_train_diagnostic_noise_chunks.scp $data_dir/orign_train_diagnostic_noise_chunks.scp
fi

echo "$0: Generate the egs for train dataset."

num_egs=$(cat $data_dir/orign_train_chunks.all.scp | wc -l)
num_archives=$[$num_egs/$egs_per_iter+1]
echo $num_archives > $egs_dir/info/num_archives

if [ -e $egs_dir/storage ]; then
  echo "$0:creating data links"
  utils/create_data_link.pl $(for x in $(seq $num_archives); do echo $egs_dir/egs.$x.ark; done)
  utils/create_data_link.pl $(for x in $(seq $num_archives); do echo $egs_dir/egs.noise.$x.ark; done)
fi

if [ $stage -le 3 ]; then
  echo "$0:shuffle and recombine train set"
  egs_scp_list=
  for n in $(seq $num_archives); do
    egs_scp_list="$egs_scp_list $egs_dir/egs.$n.scp.tmp"
  done
  utils/shuffle_list.pl $egs_dir/orign_train_chunks.all.scp > $egs_dir/orign_train_chunks.all.scp.shuffled
  utils/split_scp.pl $egs_dir/orign_train_chunks.all.scp.shuffled $egs_scp_list
  
  utils/shuffle_list.pl $egs_dir/orign_train_noise_chunks.all.scp > $egs_dir/orign_train_noise_chunks.all.scp.shuffled
  count=0
  for n in $(seq $num_archives); do
    current_count=$(cat $egs_dir/egs.$n.scp.tmp | wc -l)
    count=$[$count+2*$current_count]
    cat $egs_dir/orign_train_noise_chunks.all.scp.shuffled | head -n $count | tail -n $[2*$current_count] > $egs_dir/egs.noise.$n.scp.tmp
  done
  $cmd JOB=1:$num_archives $egs_dir/log/get_egs.JOB.log \
    copy-vector scp:$egs_dir/egs.JOB.scp.tmp ark,scp:$egs_dir/egs.JOB.ark,$egs_dir/egs.JOB.scp || exit 1;
  $cmd JOB=1:$num_archives $egs_dir/log/get_egs_noise.JOB.log \
    copy-vector scp:$egs_dir/egs.noise.JOB.scp.tmp ark,scp:$egs_dir/egs.noise.JOB.ark,$egs_dir/egs.noise.JOB.scp || exit 1;
fi

if [ $stage -le 4 ]; then
  echo "$0:shuffle and recombine valid set"
  $cmd $egs_dir/log/get_egs_valid.log \
    copy-vector scp:$egs_dir/orign_valid_chunks.scp ark,scp:$egs_dir/valid_diagnostic_egs.1.ark,$egs_dir/valid_diagnostic_egs.1.scp || exit 1;
  $cmd $egs_dir/log/get_egs_valid_noise.log \
    copy-vector scp:$egs_dir/orign_valid_noise_chunks.scp ark,scp:$egs_dir/valid_diagnoistic_egs.noise.1.ark,$egs_dir/valid_diagnostic_egs.noise.1.scp || exit 1;
fi

if [ $stage -le 5 ]; then
  echo "$0:shuffle and recombine train_diagnostic set"
  $cmd $egs_dir/log/get_egs_train_diagnostic.log \
    copy-vector scp:$egs_dir/orign_train_diagnostic_chunks.scp ark,scp:$egs_dir/train_diagnostic_egs.1.ark,$egs_dir/train_diagnostic_egs.1.scp || exit 1;
  $cmd $egs_dir/log/get_egs_train_diagnostic_noise.log \
    copy-vector scp:$egs_dir/orign_train_diagnostic_noise_chunks.scp ark,scp:$egs_dir/train_diagnostic_egs.noise.1.ark,$egs_dir/train_diagnostic_egs.noise.1.scp || exit 1;
  echo "1" > $egs_dir/info/num_diagnostic_archives
fi
echo "$0: Finished preparing fvector training examples"
