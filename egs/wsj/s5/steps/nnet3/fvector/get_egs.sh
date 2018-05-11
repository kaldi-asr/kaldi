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
                       # In xvector setup, this item is 2 milion and each frame
                       # is 40 dims. In fvector case, the dimension is about
                       # 1egs=100ms=2 * 8frames* (16kHz * 25ms)= 6400.
                       # So (2milion * 40 / 6400)
                       # If frame-length=10ms, it should be 30000.
                       # That means we keep the capacity of fvector with xvector.
egs_per_iter_diagnostic=10000    # have this many frames per achive for the
                                 # archives used for diagnostics.
num_diagnostic_percent=5   # we want to test the training and validation likelihoods
                           # on a range of utterance lengths, and this number
                           # controls how many archives we evaluate on. Select
                           # "num_diagnostic_percent"% train data to be valid
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
    fvector-chunk --chunk-size=120 "scp:utils/filter_scp.pl --exclude $egs_dir/info/valid_uttlist $sdata/JOB/wav.scp |" \
      scp:$noise_dir/wav.scp $noise_dir/utt2dur_fix \
      ark,scp:$egs_dir/orign_train_chunks.JOB.ark,$egs_dir/orign_train_chunks.JOB.scp
  for n in $(seq $nj); do
    cat $egs_dir/orign_train_chunks.${n}.scp || exit 1;
  done > $data_dir/orign_train_chunks.all.scp

  $cmd $egs_dir/log/cut_valid_wav_into_chunks.log \
    fvector-chunk --chunk-size=120 "scp:utils/filter_scp.pl $egs_dir/info/valid_uttlist $data_dir/wav.scp |" \
      scp:$noise_dir/wav.scp $noise_dir/utt2dur_fix \
      ark,scp:$egs_dir/orign_valid_chunks.ark,$egs_dir/orign_valid_chunks.scp
  cp $egs_dir/orign_valid_chunks.scp $data_dir/orign_valid_chunks.scp

  $cmd $egs_dir/log/cut_train_diagnostic_wav_into_chunks.log \
    fvector-chunk --chunk-size=120 "scp:utils/filter_scp.pl $egs_dir/info/train_diagnostic_uttlist $data_dir/wav.scp |" \
      scp:$noise_dir/wav.scp $noise_dir/utt2dur_fix \
      ark,scp:$egs_dir/orign_train_diagnostic_chunks.ark,$egs_dir/orign_train_diagnostic_chunks.scp
  cp $egs_dir/orign_train_diagnostic_chunks.scp $data_dir/orign_train_diagnostic_chunks.scp
fi

echo "$0: Generate the egs for train dataset."

#each chunk will generate two "NnetIo"s
num_egs=$(cat $data_dir/orign_train_chunks.all.scp | wc -l)
num_archives=$[$num_egs/$egs_per_iter+1]
# We may have to first create a smaller number of larger archives, with number
# $num_archives_intermediate, if $num_archives is more than the maximum number
# of open filehandles that the system allows per process (ulimit -n).
# This sometimes gives a misleading answer as GridEngine sometimes changes that
# somehow, so we limit it to 512.
max_open_filehandles=$(ulimit -n) || exit 1
[ $max_open_filehandles -gt 512 ] && max_open_filehandles=512
num_archives_intermediate=$num_archives
archives_multiple=1
while [ $[$num_archives_intermediate+4] -gt $max_open_filehandles ]; do
  archives_multiple=$[$archives_multiple+1]
  num_archives_intermediate=$[$num_archives/$archives_multiple+1];
done
# now make sure num_archives is an exact multiple of archives_multiple.
num_archives=$[$archives_multiple*$num_archives_intermediate]
echo $num_archives > $egs_dir/info/num_archives

# prepare the dir link
if [ -e $egs_dir/storage ]; then
  # Make soft links to storage directories, if distributing this way..  See
  # utils/create_split_dir.pl.
  echo "$0: creating data links"
  utils/create_data_link.pl $(for x in $(seq $num_archives); do echo $egs_dir/egs.$x.ark; done)
  for x in $(seq $num_archives_intermediate); do
    utils/create_data_link.pl $(for y in $(seq $nj); do echo $egs_dir/egs_orig.$y.$x.ark; done)
  done
fi
# Deal with the chunk one-by-one, add the noise.
# convert the chunk data into Nnet3eg
if [ $stage -le 3 ]; then
  # create egs_orig.*.*.ark; the first index goes to $nj,
  # the second to $num_archives_intermediate.
  egs_list=
  for n in $(seq $num_archives_intermediate); do
    egs_list="$egs_list ark:$egs_dir/egs_orig.JOB.$n.ark"
  done
  echo "$0: Do data perturbation and dump on disk"
  #The options could be added in this line
  $cmd JOB=1:$nj $egs_dir/log/do_train_perturbation_and_get_egs.JOB.log \
    fvector-add-noise --max-snr=20 --min-snr=10 scp:$egs_dir/orign_train_chunks.JOB.scp ark:- \| \
    fvector-get-egs ark:- ark:- \| \
    nnet3-copy-egs --random=true --srand=\$[JOB+$srand] ark:- $egs_list || exit 1;
fi

# The num_archives_intermediate looks like a bridge. It used to convert the
# egs_orig(nj * num_achives_intermediate) to egs(num_achives_intermediate * archives_multiple)
# Each time, get a colmn from egs_orig and average dispersion to a row of egs.
if [ $stage -le 4 ]; then
  echo "$0: recombining and shuffling order of archives on disk"
  # combine all the "egs_orig.*.JOB.scp" (over the $nj splits of the data) and
  # shuffle the order, writing to the egs.JOB.ark

  # the input is a concatenation over the input jobs.
  egs_list=
  for n in $(seq $nj); do
    egs_list="$egs_list $egs_dir/egs_orig.$n.JOB.ark"
  done

  if [ $archives_multiple == 1 ]; then # normal case.
    if $generate_egs_scp; then
      output_archive="ark,scp:$egs_dir/egs.JOB.ark,$egs_dir/egs.JOB.scp"
    else
      output_archive="ark:$egs_dir/egs.JOB.ark"
    fi
    $cmd --max-jobs-run $nj JOB=1:$num_archives_intermediate $egs_dir/log/shuffle.JOB.log \
      nnet3-shuffle-egs --srand=\$[JOB+$srand] "ark:cat $egs_list|" $output_archive  || exit 1;

    if $generate_egs_scp; then
      #concatenate egs.JOB.scp in single egs.scp
      rm $egs_dir/egs.scp 2> /dev/null || true
      for j in $(seq $num_archives_intermediate); do
        cat $egs_dir/egs.$j.scp || exit 1;
      done > $egs_dir/egs.scp || exit 1;
      for f in $egs_dir/egs.*.scp; do rm $f; done
    fi
  else
    # we need to shuffle the 'intermediate archives' and then split into the
    # final archives.  we create soft links to manage this splitting, because
    # otherwise managing the output names is quite difficult (and we don't want
    # to submit separate queue jobs for each intermediate archive, because then
    # the --max-jobs-run option is hard to enforce).
    if $generate_egs_scp; then
      output_archives="$(for y in $(seq $archives_multiple); do echo ark,scp:$egs_dir/egs.JOB.$y.ark,$egs_dir/egs.JOB.$y.scp; done)"
    else
      output_archives="$(for y in $(seq $archives_multiple); do echo ark:$egs_dir/egs.JOB.$y.ark; done)"
    fi
    for x in $(seq $num_archives_intermediate); do
      for y in $(seq $archives_multiple); do
        archive_index=$[($x-1)*$archives_multiple+$y]
        # egs.intermediate_archive.{1,2,...}.ark will point to egs.archive.ark
        ln -sf egs.$archive_index.ark $egs_dir/egs.$x.$y.ark || exit 1
      done
    done
    $cmd --max-jobs-run $nj JOB=1:$num_archives_intermediate $egs_dir/log/shuffle.JOB.log \
      nnet3-shuffle-egs --srand=\$[JOB+$srand] "ark:cat $egs_list|" ark:- \| \
      nnet3-copy-egs ark:- $output_archives || exit 1;

    if $generate_egs_scp; then
      #concatenate egs.JOB.scp in single egs.scp
      rm $egs_dir/egs.scp 2> /dev/null || true
      for j in $(seq $num_archives_intermediate); do
        for y in $(seq $num_archives_intermediate); do
          cat $egs_dir/egs.$j.$y.scp || exit 1;
        done
      done > $egs_dir/egs.scp || exit 1;
      for f in $egs_dir/egs.*.*.scp; do rm $f; done
    fi
  fi
fi
#get egs.$archives_multiple.$num_archives_intermediate.ark 
#get egs.scp

echo "$0: Generate the egs for valid dataset"
if [ $stage -le 5 ]; then
  $cmd $egs_dir/log/do_valid_perturbation_and_get_egs.log \
    fvector-add-noise --max-snr=20 --min-snr=10 scp:$egs_dir/orign_valid_chunks.scp ark:- \| \
    fvector-get-egs ark:- ark:- \| \
    nnet3-copy-egs --random=true --srand=$srand ark:- ark:$egs_dir/valid.egs || exit 1;
  #get the valid.egs
  cp $egs_dir/valid.egs $egs_dir/valid_diagnostic_egs.1.ark
fi

echo "$0: Generate the egs for train diagnostic"
if [ $stage -le 6 ];then
  $cmd $egs_dir/log/do_train_diagnostic_perturbation_and_get_egs.log \
    fvector-add-noise --max-snr=20 --min-snr=10 scp:$egs_dir/orign_train_diagnostic_chunks.scp ark:- \| \
    fvector-get-egs ark:- ark:- \| \
    nnet3-copy-egs --random=true --srand=$srand ark:- ark:$egs_dir/train_diagnostic.egs || exit 1;
  #get the train_diagnostic.egs
  cp $egs_dir/train_diagnostic.egs $egs_dir/train_diagnostic_egs.1.ark
  echo "1" > $egs_dir/info/num_diagnostic_archives
fi

# remove unnecessary arks and links.
if [ $stage -le 7 ]; then
  echo "$0: removing temporary archives"
  for x in $(seq $nj); do
    for y in $(seq $num_archives_intermediate); do
      file=$egs_dir/egs_orig.$x.$y.ark
      [ -L $file ] && rm $(utils/make_absolute.sh $file)
      rm $file
    done
  done
  if [ $archives_multiple -gt 1 ]; then
    # there are some extra soft links that we should delete.
    for f in $egs_dir/egs.*.*.ark; do rm $f; done
  fi
fi
echo "$0: Finished preparing fvector training examples"
