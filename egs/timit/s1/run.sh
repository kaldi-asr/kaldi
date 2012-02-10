. path.sh
local/timit_data_prep.sh /ais/gobi2/speech/TIMIT
local/timit_train_lms.sh data/local
local/timit_format_data.sh

# mfccdir should be some place with a largish disk where you
# want to store MFCC features. 
mfccdir=mfccs

steps/make_mfcc.sh data/train exp/make_mfcc/train $mfccdir 4
for test in train test dev ; do
  steps/make_mfcc.sh data/$test exp/make_mfcc/$test $mfccdir 4
done

# train monophone system.
steps/train_mono.sh data/train data/lang exp/mono

scripts/mkgraph.sh --mono data/lang_test exp/mono exp/mono/graph
echo "Decoding test datasets."
for test in dev test ; do
  steps/decode_deltas.sh exp/mono data/$test data/lang exp/mono/decode_$test &
done
wait
scripts/average_wer.sh exp/mono/decode_*/wer > exp/mono/wer

# Get alignments from monophone system.
echo "Creating training alignments to use to train other systems such as ANN-HMM."
steps/align_deltas.sh data/train data/lang exp/mono exp/mono_ali
echo "Creating dev alignments to use to train other systems such as ANN-HMM."
steps/align_deltas.sh data/dev data/lang exp/mono exp/mono_ali_dev


