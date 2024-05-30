#!/usr/bin/env bash
# Copyright 2015   David Snyder
# Apache 2.0.
#
if [ $# != 6 ]; then
  echo "Usage: $0 <plda-data-dir> <enroll-data-dir> <test-data-dir> <plda-ivec-dir> <enroll-ivec-dir> <test-ivec-dir>"
fi
plda_data_dir=${1%/}
enroll_data_dir=${2%/}
test_data_dir=${3%/}
plda_ivec_dir=${4%/}
enroll_ivec_dir=${5%/}
test_ivec_dir=${6%/}

if [ ! -f ${test_data_dir}/trials ]; then
  echo "${test_data_dir} needs a trial file."
  exit;
fi

mkdir -p local/.tmp

# Partition the SRE data into male and female subsets.
cat ${test_data_dir}/spk2gender | grep -w f > local/.tmp/female_spklist
utils/subset_data_dir.sh --spk-list local/.tmp/female_spklist ${test_data_dir} ${test_data_dir}_female
cat ${enroll_data_dir}/spk2gender | grep -w f > local/.tmp/female_spklist
utils/subset_data_dir.sh --spk-list local/.tmp/female_spklist ${enroll_data_dir} ${enroll_data_dir}_female
cat ${test_data_dir}/spk2gender | grep -w m > local/.tmp/male_spklist
utils/subset_data_dir.sh --spk-list local/.tmp/male_spklist ${test_data_dir} ${test_data_dir}_male
cat ${enroll_data_dir}/spk2gender | grep -w m > local/.tmp/male_spklist
utils/subset_data_dir.sh --spk-list local/.tmp/male_spklist ${enroll_data_dir} ${enroll_data_dir}_male
cat ${plda_data_dir}/spk2gender | grep -w f > local/.tmp/female_spklist
utils/subset_data_dir.sh --spk-list local/.tmp/female_spklist ${plda_data_dir} ${plda_data_dir}_female
cat ${plda_data_dir}/spk2gender | grep -w m > local/.tmp/male_spklist
utils/subset_data_dir.sh --spk-list local/.tmp/male_spklist ${plda_data_dir} ${plda_data_dir}_male

# Prepare female and male trials.
trials_female=${test_data_dir}_female/trials
cat ${test_data_dir}/trials | awk '{print $2, $0}' | \
  utils/filter_scp.pl ${test_data_dir}_female/utt2spk | cut -d ' ' -f 2- \
  > $trials_female
trials_male=${test_data_dir}_male/trials
cat ${test_data_dir}/trials | awk '{print $2, $0}' | \
  utils/filter_scp.pl ${test_data_dir}_male/utt2spk | cut -d ' ' -f 2- \
  > $trials_male

mkdir -p ${test_ivec_dir}_male
mkdir -p ${test_ivec_dir}_female
mkdir -p ${enroll_ivec_dir}_male
mkdir -p ${enroll_ivec_dir}_female
mkdir -p ${plda_ivec_dir}_male
mkdir -p ${plda_ivec_dir}_female

# Partition the i-vectors into male and female subsets.
utils/filter_scp.pl ${enroll_data_dir}_male/utt2spk \
  ${enroll_ivec_dir}/ivector.scp > ${enroll_ivec_dir}_male/ivector.scp
utils/filter_scp.pl ${test_data_dir}_male/utt2spk \
  ${test_ivec_dir}/ivector.scp > ${test_ivec_dir}_male/ivector.scp
utils/filter_scp.pl ${enroll_data_dir}_female/utt2spk \
  ${enroll_ivec_dir}/ivector.scp > ${enroll_ivec_dir}_female/ivector.scp
utils/filter_scp.pl ${test_data_dir}_female/utt2spk \
  ${test_ivec_dir}/ivector.scp > ${test_ivec_dir}_female/ivector.scp
utils/filter_scp.pl ${plda_data_dir}_female/utt2spk \
  ${plda_ivec_dir}/ivector.scp > ${plda_ivec_dir}_female/ivector.scp
utils/filter_scp.pl ${plda_data_dir}_male/utt2spk \
  ${plda_ivec_dir}/ivector.scp > ${plda_ivec_dir}_male/ivector.scp
utils/filter_scp.pl ${enroll_data_dir}_male/spk2utt \
  ${enroll_ivec_dir}/spk_ivector.scp > ${enroll_ivec_dir}_male/spk_ivector.scp
utils/filter_scp.pl ${enroll_data_dir}_female/spk2utt \
  ${enroll_ivec_dir}/spk_ivector.scp > ${enroll_ivec_dir}_female/spk_ivector.scp
utils/filter_scp.pl ${enroll_data_dir}_male/spk2utt \
  ${enroll_ivec_dir}/num_utts.ark > ${enroll_ivec_dir}_male/num_utts.ark
utils/filter_scp.pl ${enroll_data_dir}_female/spk2utt \
  ${enroll_ivec_dir}/num_utts.ark > ${enroll_ivec_dir}_female/num_utts.ark
utils/filter_scp.pl ${plda_data_dir}_male/spk2utt \
  ${plda_ivec_dir}/spk_ivector.scp > ${plda_ivec_dir}_male/spk_ivector.scp
utils/filter_scp.pl ${plda_data_dir}_female/spk2utt \
  ${plda_ivec_dir}/spk_ivector.scp > ${plda_ivec_dir}_female/spk_ivector.scp
utils/filter_scp.pl ${plda_data_dir}_male/spk2utt \
  ${plda_ivec_dir}/num_utts.ark > ${plda_ivec_dir}_male/num_utts.ark
utils/filter_scp.pl ${plda_data_dir}_female/spk2utt \
  ${plda_ivec_dir}/num_utts.ark > ${plda_ivec_dir}_female/num_utts.ark

# Compute gender independent and dependent i-vector means.
run.pl ${plda_ivec_dir}/log/compute_mean.log \
  ivector-normalize-length scp:${plda_ivec_dir}/ivector.scp \
  ark:- \| ivector-mean ark:- ${plda_ivec_dir}/mean.vec || exit 1;
run.pl ${plda_ivec_dir}_male/log/compute_mean.log \
  ivector-normalize-length scp:${plda_ivec_dir}_male/ivector.scp \
  ark:- \| ivector-mean ark:- ${plda_ivec_dir}_male/mean.vec || exit 1;
run.pl ${plda_ivec_dir}_female/log/compute_mean.log \
  ivector-normalize-length scp:${plda_ivec_dir}_female/ivector.scp \
  ark:- \| ivector-mean ark:- ${plda_ivec_dir}_female/mean.vec || exit 1;

rm -rf local/.tmp
