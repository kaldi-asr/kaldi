data=$1

if [ ! -s $data/segments ]; then
  utils/data/get_segments_for_data.sh $data > $data/segments
fi

cut -d ' ' -f 1,2 $data/segments | utils/utt2spk_to_spk2utt.pl > $data/reco2utt
