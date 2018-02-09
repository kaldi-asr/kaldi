#!/bin/bash

stage=0
cmd=run.pl
nj=60
data=data/swbd_sre_combined_no_sil
dir=exp/xvector_nnet/egs

#some options
num_heldout_utts=1000       # number of utterances held out for training subset
                            # and validation set
min_frames_per_chunk=200
max_frames_per_chunk=400

kinds_of_length_train=100   # number of the kinds of lengths in train set
kinds_of_length_valid=3     # number of the kinds of lengths in valiation set
                            # and training subset

frames_per_iter=70000000    # target number of frames per archive. If it is null,
                            # each new directory corresponds to an archive
distinct_length_ark=false   # If true, all the egs will combine and shuffle; if false
                            # only the same length egs will be allocated to the same
                            # archive. Otherwise, if "frames_per_iter" is supplied,
                            # each kind of length egs will average few archives, otherwise 
                            # the egs which has the same length will store in one archive.
num_egs_per_speaker=5000    # number of egs for each speaker
average_method=false
#end of options

# Something need to be mentioned. When we allocate the whole egs into different
# archives, we provide three ways. You can choose what you like according to
# assign "distinct_length_ark" and "frames_per_iter" options.
# (1) If the "distinct_length_ark" is true, you must set a value to "frames_per_iter"
# and it means the egs in the same archive will have distinct lengths. We will
# count the frames and allocate them into pieces in average.
# If the "distinct_length_ark" is false, you can set a value to "frames_per_iter" or not.
# (2) If the "distinct_length_ark" is false and "frames_per_iter" is set. It means
# in the same archive, the length of egs is same. And the capacity of certain 
# archive will be smaller than "frames_per_iter". We will count the amount of
# frames of each kind of length, and allocate them into pieces in average.
# (3) If the "distinct_length_ark" is false and "frames_per_iter" is not set.
# It means all the data which has the same length will be assign to one archive.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 2 ]; then
  echo "Usage: $0 [opts] <data> <egs-dir>"
  echo " e.g.: $0 data/train exp/xvector_a/egs"
  echo ""
  echo "Main options (for others, see top of script file)"
  echo "  --config <config-file>                           # config file containing options"
  echo "  --nj <nj>                                        # The maximum number of jobs you want to run in"
  echo "                                                   # parallel (increase this only if you have good disk and"
  echo "                                                   # network speed).  default=6"
  echo "  --cmd (utils/run.pl;utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --min-frames-per-eg <#frames;50>                 # The minimum number of frames per chunk that we dump"
  echo "  --max-frames-per-eg <#frames;200>                # The maximum number of frames per chunk that we dump"
  echo "  --frames-per-iter <#samples;1000000>             # Target number of frames per archive"
  echo "  --num-diagnostic-archives <#archives;3>          # Option that controls how many different versions of"
  echo "                                                   # the train and validation archives we create (e.g."
  echo "                                                   # train_subset.{1,2,3}.egs and valid.{1,2,3}.egs by default;"
  echo "                                                   # they contain different utterance lengths."
  echo "  --stage <stage|0>                                # Used to run a partially-completed training process from somewhere in"
  echo "                                                   # the middle."
  exit 1;
fi

data=$1
dir=$2

for f in $data/utt2spk $data/utt2num_frames $data/feats.scp ; do
  [ ! -f $f ] && echo "$0: expected file $f" && exit 1;
done

num_egs_per_speaker_per_length=$[$num_egs_per_speaker/$kinds_of_length_train]
feat_dim=$(feat-to-dim scp:$data/feats.scp -) || exit 1;

mkdir -p $dir/log $dir/info $dir/temp $dir/train_set/temp $dir/valid_set/temp $dir/train_subset/temp
temp=$dir/temp

echo $feat_dim > $dir/info/feat_dim
echo '0' > $dir/info/left_context
echo $min_frames_per_chunk > $dir/info/right_context
echo '1' > $dir/info/frames_per_eg

if [ $stage -le 0 ]; then
  echo "$0: Preparing train and validation lists"
  # Pick a list of heldout utterances for validation egs
  cat $data/utt2spk | utils/shuffle_list.pl | head -$num_heldout_utts > $dir/valid_set/utt2spk || exit 1;
  cp $dir/valid_set/utt2spk $temp/uttlist_valid
  utils/filter_scp.pl $dir/valid_set/utt2spk $data/utt2num_frames > $dir/valid_set/utt2num_frames
  utils/filter_scp.pl $dir/valid_set/utt2spk $data/feats.scp > $dir/valid_set/feats.scp

  # The remaining utterances are used for training egs
  utils/filter_scp.pl --exclude $dir/valid_set/utt2spk $data/utt2spk > $dir/train_set/utt2spk
  cp $dir/train_set/utt2spk $temp/uttlist_train
  utils/filter_scp.pl --exclude $dir/valid_set/utt2spk $data/utt2num_frames > $dir/train_set/utt2num_frames
  utils/filter_scp.pl --exclude $dir/valid_set/utt2spk $data/feats.scp > $dir/train_set/feats.scp

  # Pick a subset of the training list for diagnostics
  cat $dir/train_set/utt2spk | utils/shuffle_list.pl | head -$num_heldout_utts > $dir/train_subset/utt2spk || exit 1;
  cp $dir/train_subset/utt2spk $temp/uttlist_train_subset
  utils/filter_scp.pl $temp/uttlist_train_subset <$data/utt2num_frames > $dir/train_subset/utt2num_frames
  utils/filter_scp.pl $temp/uttlist_train_subset <$data/feats.scp > $dir/train_subset/feats.scp

  # Create a mapping from utterance to speaker ID (an integer)
  awk -v id=0 '{print $1, id++}' $data/spk2utt > $temp/spk2int
  utils/sym2int.pl -f 2 $temp/spk2int $data/utt2spk > $temp/utt2int
  utils/filter_scp.pl $dir/train_set/utt2spk $temp/utt2int > $dir/train_set/utt2int
  utils/filter_scp.pl $dir/valid_set/utt2spk $temp/utt2int > $dir/valid_set/utt2int
  utils/filter_scp.pl $dir/train_subset/utt2spk $temp/utt2int > $dir/train_subset/utt2int
  #Above, prepare the "utt2num_frames, utt2spk and utt2int" for each data set.
fi

num_pdfs=$(awk '{print $2}' $temp/utt2int | sort | uniq -c | wc -l)
num_train_set_frames=$(awk '{n += $2} END{print n}' <$dir/train_set/utt2num_frames)
num_train_subset_frames=$(awk '{n += $2} END{print n}' <$dir/train_subset/utt2num_frames)
echo $num_train_set_frames > $dir/info/num_frames
echo $num_pdfs > $dir/info/num_pdfs

num_egs_per_speaker_per_length=$[$num_egs_per_speaker/$kinds_of_length_train]
echo $kinds_of_length_train > $dir/info/kinds_of_length
echo $num_egs_per_speaker_per_length > $dir/info/num_egs_per_speaker_per_length

temp_dir=$dir/train_set/temp
if [ $stage -le 1 ]; then
  echo "$0: Allocating training egs"
  #Generate data_li/feat.scp and utt2spk
  $cmd $dir/log/allocate_examples.log \
    sid/nnet3/xvector/allocate_egs_seg.py \
      --min-frames-per-chunk=$min_frames_per_chunk \
      --max-frames-per-chunk=$max_frames_per_chunk \
      --kinds-of-length=$kinds_of_length_train \
      --generate-validate=false \
      --randomize-chunk-length=true \
      --num-egs-per-speaker-per-length=$num_egs_per_speaker_per_length \
      --average-method=$average_method \
      --data-dir=$dir/train_set \
      --output-dir=$temp_dir || exit 1
  #sort and uniq
  for subdir in `dir $temp_dir`; do
    utils/fix_data_dir.sh $temp_dir/$subdir
    utils/sym2int.pl -f 2 $dir/temp/spk2int $temp_dir/$subdir/utt2spk \
      > $temp_dir/$subdir/utt2int
    this_length=`echo $subdir | cut -d_ -f 2`
    this_num_utts=$(cat $temp_dir/$subdir/feats.scp | wc -l)
    this_frames=$[$this_length*$this_num_utts]
    echo "$this_frames" > $temp_dir/$subdir/num_frames
  done
fi

num_train_archives=0

if [ $stage -le 2 ];then
  echo "$0: Combine, shuffle and split list for training set"
  #the lengths of egs in the same archive are same.
  if [ -e $temp_dir/combine ] && rm -rf $temp_dir/combine
  dir_list=`dir $temp_dir | shuf`
  if [ $distinct_length_ark ==  "false" ]; then 
    
    #combine feats.scp
    mkdir -p $temp_dir/combine
    data_combine=$temp_dir/combine

    utils/combine_data.sh $data_combine $temp_dir/data_*
    utils/sym2int.pl -f 2 $dir/temp/spk2int ${data_combine}/utt2spk \
      > ${data_combine}/utt2int

    if [ -z "$frames_per_iter" ]; then
      # each new directory corresponds to an archive, shuffle the order so that
      # the order of archives is not same with the order of `dir $temp_dir`
      for subdir in $dir_list; do
        num_train_archives=$[$num_train_archives+1]
        cp $temp_dir/$subdir/feats.scp ${data_combine}/shuffled_feats.${num_train_archives}.scp
      done
    else 
      # the option "frames_per_iter" limits the capacity of each archive
      for subdir in $dir_list; do
        current_frames=`cat $temp_dir/$subdir/num_frames`
        current_archives=$[$current_frames/$frames_per_iter+1]
        # split feats.scp
        feat_scps=$(for n in `seq $[$num_train_archives+1] $[$num_train_archives+$current_archives]`; do echo ${data_combine}/shuffled_feats.${n}.scp; done)
        utils/split_scp.pl $temp_dir/$subdir/feats.scp $feat_scps
        num_train_archives=$[$num_train_archives+$current_archives]
      done
    fi
  else 
    if [ -z "$frames_per_iter" ]; then
      echo "For distinct length egs in one archive method, we need option --frames-per-iter"
    fi
    num_train_frames=0
    for subdir in $dir_list; do
      current_frames=`cat $temp_dir/$subdir/num_frames`
      num_train_frames=$[$num_train_frames+$current_frames]
    done
    num_train_archives=$[$num_train_frames/$frames_per_iter+1]

    #combine feats.scp and shuffle
    mkdir -p $temp_dir/combine
    data_combine=$temp_dir/combine

    utils/combine_data.sh $data_combine $temp_dir/data_*
    utils/shuffle_list.pl $data_combine/feats.scp > $data_combine/shuffled_feats.scp
    utils/sym2int.pl -f 2 $dir/temp/spk2int ${data_combine}/utt2spk \
      > ${data_combine}/utt2int

    feat_scps=$(for n in `seq $num_train_archives`; do echo $data_combine/shuffled_feats.${n}.scp; done)
    utils/split_scp.pl ${data_combine}/shuffled_feats.scp $feat_scps
  fi
fi

if [ $stage -le 3 ]; then
  echo "$0: Dump Egs for training set"
  if [ -e $dir/storage ]; then
    echo "$0: creating data links"
    utils/create_data_link.pl $(for x in $(seq $num_train_archives); do echo $dir/egs.$x.ark; done)
  fi
  #Note: if --num-pdfs options is not supply, you must use global utt2int,
  #because we will compute the num-pdfs from utt2label file.
  $cmd --max-jobs-run $nj JOB=1:$num_train_archives $dir/log/dump_egs.JOB.log \
    nnet3-xvector-get-egs-seg \
      --num-pdfs=$num_pdfs \
      $temp_dir/combine/utt2int \
      scp:$temp_dir/combine/shuffled_feats.JOB.scp \
      ark,scp:$dir/egs.JOB.ark,$dir/egs.JOB.scp
fi

temp_dir=$dir/valid_set/temp
if [ $stage -le 4 ]; then
  echo "$0: Dealing with validation set"
  #Generate data_li/feat.scp and utt2spk
  $cmd $dir/log/allocate_examples_validate.log \
    sid/nnet3/xvector/allocate_egs_seg.py \
      --min-frames-per-chunk=$min_frames_per_chunk \
      --max-frames-per-chunk=$max_frames_per_chunk \
      --kinds-of-length=$kinds_of_length_valid \
      --generate-validate=true \
      --randomize-chunk-length=false \
      --data-dir=$dir/valid_set \
      --output-dir=$temp_dir || exit 1
  #Generate data_li/spk2utt and utt2label
  for subdir in `dir $temp_dir`; do
    utils/fix_data_dir.sh $temp_dir/$subdir
    utils/sym2int.pl -f 2 $dir/temp/spk2int $temp_dir/$subdir/utt2spk \
      > $temp_dir/$subdir/utt2int
  done

  echo "$0: combine and shuffle (valid)"
  #combine feats.scp and shuffle
  mkdir -p $temp_dir/combine
  data_combine=$temp_dir/combine

  utils/combine_data.sh $data_combine $temp_dir/data_*
  utils/shuffle_list.pl $data_combine/feats.scp > $data_combine/shuffled_feats.scp
  utils/sym2int.pl -f 2 $dir/temp/spk2int ${data_combine}/utt2spk \
    > ${data_combine}/utt2int

  feat_scps=$(for n in `seq $kinds_of_length_valid`; do echo $data_combine/shuffled_feats.${n}.scp; done)
  utils/split_scp.pl ${data_combine}/shuffled_feats.scp $feat_scps

  echo "$0: dump egs (valid)"
  #dump egs
  $cmd --max-jobs-run $nj JOB=1:$kinds_of_length_valid $dir/log/dump_egs_valid.JOB.log \
    nnet3-xvector-get-egs-seg \
      --num-pdfs=$num_pdfs \
      $temp_dir/combine/utt2int \
      scp:$temp_dir/combine/shuffled_feats.JOB.scp \
      ark,scp:$dir/valid_egs.JOB.ark,$dir/valid_egs.JOB.scp
  cat $dir/valid_egs.*.scp > $dir/valid_diagnostic.scp
fi

temp_dir=$dir/train_subset/temp
if [ $stage -le 5 ]; then
  echo "$0: Dealing with train_diagnostic set"
  #Generate data_li/feat.scp and utt2spk
  $cmd $dir/log/allocate_examples_diagnostic.log \
    sid/nnet3/xvector/allocate_egs_seg.py \
      --min-frames-per-chunk=$min_frames_per_chunk \
      --max-frames-per-chunk=$max_frames_per_chunk \
      --kinds-of-length=$kinds_of_length_valid \
      --generate-validate=true \
      --randomize-chunk-length=false \
      --data-dir=$dir/train_subset \
      --output-dir=$temp_dir || exit 1
  #Generate data_li/spk2utt and utt2label
  for subdir in `dir $temp_dir`; do
    utils/fix_data_dir.sh $temp_dir/$subdir
    utils/sym2int.pl -f 2 $dir/temp/spk2int $temp_dir/$subdir/utt2spk \
      > $temp_dir/$subdir/utt2int
  done

  echo "$0: combine and shuffle (diagnostic)"
  #combine feats.scp and shuffle
  mkdir -p $temp_dir/combine
  data_combine=$temp_dir/combine

  utils/combine_data.sh $data_combine $temp_dir/data_*
  utils/shuffle_list.pl $data_combine/feats.scp > $data_combine/shuffled_feats.scp
  utils/sym2int.pl -f 2 $dir/temp/spk2int ${data_combine}/utt2spk \
    > ${data_combine}/utt2int

  feat_scps=$(for n in `seq $kinds_of_length_valid`; do echo $data_combine/shuffled_feats.${n}.scp; done)
  utils/split_scp.pl ${data_combine}/shuffled_feats.scp $feat_scps

  echo "$0: dump egs (diagnostic)"
  #dump egs
  $train_cmd --max-jobs-run $nj JOB=1:$kinds_of_length_valid $dir/log/dump_egs_diagnostic.JOB.log \
    nnet3-xvector-get-egs-seg \
      --num-pdfs=$num_pdfs \
      $temp_dir/combine/utt2int \
      scp:$temp_dir/combine/shuffled_feats.JOB.scp \
      ark,scp:$dir/train_diagnostic_egs.JOB.ark,$dir/train_diagnostic_egs.JOB.scp
  cat $dir/train_diagnostic_egs.*.scp > $dir/train_diagnostic.scp
  if [ -e $dir/combine.scp ]; then
    rm -rf $dir/combine.scp
    ln -s $dir/train_diagnostic.scp $dir/combine.scp
  fi
fi
