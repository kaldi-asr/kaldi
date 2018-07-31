#This script is not really supposed to be run directly
#Instead, it should be sourced from the decoding script
#It makes many assumption on existence of certain environmental
#variables as well as certain directory structure.
if [ ${dataset_type} != "supervised" ] ; then
  mandatory_variables="my_data_dir my_data_list my_nj"
  optional_variables=""
else
  mandatory_variables="my_data_dir my_data_list my_nj"
  optional_variables="my_stm_file"
fi

check_variables_are_set

segmentation_opts="--isolated-resegmentation \
  --min-inter-utt-silence-length 1.0 \
  --silence-proportion 0.05 "

workdir=exp/make_seg/${dataset_id}
unseg_dir=$workdir
mkdir -p $unseg_dir
# 4. Create the wav.scp file:
sph2pipe=`which sph2pipe || which $KALDI_ROOT/tools/sph2pipe_v2.5/sph2pipe`
if [ $? -ne 0 ] ; then
  echo "Could not find sph2pipe binary. Add it to PATH"
  exit 1;
fi
sox=`which sox`
if [ $? -ne 0 ] ; then
  echo "Could not find sox binary. Add it to PATH"
  exit 1;
fi

echo "Creating the $unseg_dir/wav.scp file"
audiodir=$my_data_dir/audio
for file in `cat $my_data_list | sort -u` ; do
  if [ -f $audiodir/$file.sph ] ; then
    echo "$file $sph2pipe -f wav -p -c 1 $audiodir/$file.sph |"
  elif [ -f $audiodir/$file.wav ] ; then
    echo "$file $sox $audiodir/$file.wav -r 8000 -c 1 -b 16 -t wav - downsample |"
  else
    echo "Audio file $audiodir/$file.(sph|wav) does not exist!" >&2
    exit 1
  fi
done | sort -u > $unseg_dir/wav.scp

l1=`cat $unseg_dir/wav.scp | wc -l `
l2=`cat $my_data_list | wc -l `
if [ "$l1" -ne "$l2" ] ; then
  echo "wav.scp number of files: $l1"
  echo "filelist number of files: $l2"
  echo "Not all files from the list $my_data_list found their way into wav.scp"
  exit 1
fi

echo "Creating the $unseg_dir/reco2file_and_channel file"
cat $unseg_dir/wav.scp | awk '{print $1, $1, "A";}' > $unseg_dir/reco2file_and_channel
cat $unseg_dir/wav.scp | awk '{print $1, $1;}' > $unseg_dir/utt2spk
utils/utt2spk_to_spk2utt.pl $unseg_dir/utt2spk > $unseg_dir/spk2utt

make_plp $unseg_dir $workdir/make_plp  $workdir/plp || exit 1

local/resegment/generate_segments.sh --nj $my_nj --cmd "$decode_cmd" \
  --noise_oov false --segmentation_opts "$segmentation_opts" \
  $unseg_dir data/lang exp/tri4b_seg \
  $workdir $dataset_dir || exit 1

num_hours=`cat ${dataset_dir}/segments | \
  awk '{secs+= $4-$3;} END{print(secs/3600);}'`

echo "Number of hours of the newly segmented data: $num_hours"

if [ "$dataset_kind" == "supervised" ]; then
  echo ---------------------------------------------------------------------
  echo "Preparing ${dataset_id} stm files in ${dataset_dir} on" `date`
  echo ---------------------------------------------------------------------
  if [ ! -z $my_stm_file ] ; then
    local/augment_original_stm.pl $my_stm_file ${dataset_dir}
  else
    local/prepare_stm.pl --fragmentMarkers \-\*\~ ${dataset_dir}
  fi
fi

