. path.sh
#local/timit_data_prep.sh /ais/gobi2/speech/TIMIT
local/timit_data_prep.sh /mnt/matylda2/data/TIMIT || exit 1;
local/timit_train_lms.sh data/local || exit 1;
local/timit_format_data.sh || exit 1;

# mfccdir should be some place with a largish disk where you
# want to store MFCC features. 
mfccdir=mfccs

steps/make_mfcc.sh data/train exp/make_mfcc/train $mfccdir 4
for test in train test dev ; do
  steps/make_mfcc.sh data/$test exp/make_mfcc/$test $mfccdir 4 || exit 1;
done

# train monophone system.
steps/train_mono.sh data/train data/lang exp/mono || exit 1;

scripts/mkgraph.sh --mono data/lang_test exp/mono exp/mono/graph || exit 1;
echo "Decoding test datasets."
for test in dev test ; do
  steps/decode_deltas.sh exp/mono data/$test data/lang exp/mono/decode_$test &
done
wait
scripts/average_wer.sh exp/mono/decode_*/wer > exp/mono/wer

# Get alignments from monophone system.
echo "Creating training alignments to use to train other systems such as ANN-HMM."
steps/align_deltas.sh data/train data/lang exp/mono exp/mono_ali || exit 1;
echo "Creating dev alignments to use to train other systems such as ANN-HMM."
steps/align_deltas.sh data/dev data/lang exp/mono exp/mono_ali_dev || exit 1;

