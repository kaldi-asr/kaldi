#!/bin/bash

#    This script normalizes hypothesis and reference file and performs scoring.
#    Eg. ./local/yomdle/normalized_scoring/normalized_score.sh <output-dir> <input-hyp-file> <slam-language>

if [ $# -ne 3 ]; then
    echo "USAGE:  ./local/yomdle/normalized_scoring/normalized_score.sh <output-dir> <input-hyp-file> <slam-language>"
    exit 1
fi

OUTDIR=$1
HYP_FILE=$2
LANG=$3

# ocr_score.pl is slow, especially for CER computation
# Therefore default option is to convert files to uxxxx format and use sclite for scoring
# Turn following switch to false to use ocr_score.pl instead
USE_SCLITE=true
script_dir=local/yomdle/normalized_scoring
OCR_SCORE=${script_dir}/ocr_score.pl
SCLITE=../../../tools/sctk/bin/sclite

LANG=$(echo $LANG | tr '[:upper:]' '[:lower:]')
echo "performing some normalizations..."

mkdir -p $OUTDIR
cat $HYP_FILE | python3 $script_dir/convert2snor.py > data/local/text/hyp_file.txt
cat data/test/text.old | python3 $script_dir/convert2snor.py > data/local/text/ref_file.txt
# Step 1. Run some normalizations that are common to all languages
python3 ${script_dir}/utils/normalize_spaces.py data/local/text/hyp_file.txt $OUTDIR/hyp.norm-sp.txt
python3 ${script_dir}/utils/normalize_spaces.py data/local/text/ref_file.txt $OUTDIR/ref.norm-sp.txt

python3 ${script_dir}/utils/normalize_common.py $OUTDIR/hyp.norm-sp.txt $OUTDIR/hyp.norm-sp-common.txt
python3 ${script_dir}/utils/normalize_common.py $OUTDIR/ref.norm-sp.txt $OUTDIR/ref.norm-sp-common.txt

# Step 1. Run language specific normalization
if [ "$LANG" == "farsi" ]; then
    # Farsi Normalization
    python3 ${script_dir}/utils/normalize_farsi.py $OUTDIR/hyp.norm-sp-common.txt $OUTDIR/hyp.norm-final.txt
    python3 ${script_dir}/utils/normalize_farsi.py $OUTDIR/ref.norm-sp-common.txt $OUTDIR/ref.norm-final.txt
else
    # For now no normalization for other langs
    cp $OUTDIR/hyp.norm-sp-common.txt $OUTDIR/hyp.norm-final.txt
    cp $OUTDIR/ref.norm-sp-common.txt $OUTDIR/ref.norm-final.txt
fi

# Step 2. Run tokenization to get word-based output
python3 ${script_dir}/utils/trans_to_tokenized_words.py $OUTDIR/hyp.norm-final.txt $OUTDIR/hyp.norm-final.words.txt
python3 ${script_dir}/utils/trans_to_tokenized_words.py $OUTDIR/ref.norm-final.txt $OUTDIR/ref.norm-final.words.txt

# Step 3. Also need to turn into space-seperated character stream to get char-based output
python3 ${script_dir}/utils/trans_to_chars.py $OUTDIR/hyp.norm-final.txt $OUTDIR/hyp.norm-final.chars.txt
python3 ${script_dir}/utils/trans_to_chars.py $OUTDIR/ref.norm-final.txt $OUTDIR/ref.norm-final.chars.txt

# Step 5. Look for reference uttids that aren't in hypothesis and add them in as blank hypotheses. This is needed because
#         otherwise sclite will not penalize systems for missing hypotheses
#python3 ${script_dir}/utils/find_missing_hyp_ids.py $OUTDIR/ref.norm-final.words.txt $OUTDIR/hyp.norm-final.words.txt > $OUTDIR/missing-hyp-ids.list
#python3 ${script_dir}/utils/insert_empty_hyp.py $OUTDIR/missing-hyp-ids.list $OUTDIR/hyp.norm-final.words.txt $OUTDIR/hyp.norm-final.words.withmissing.txt
#python3 ${script_dir}/utils/insert_empty_hyp.py $OUTDIR/missing-hyp-ids.list $OUTDIR/hyp.norm-final.chars.txt $OUTDIR/hyp.norm-final.chars.withmissing.txt

# Step 5. Look for reference uttids that aren't in hypothesis and add them in as blank hypotheses. This is needed because
#         otherwise sclite will not penalize systems for missing hypotheses
python3 ${script_dir}/utils/find_missing_hyp_ids.py $OUTDIR/ref.norm-final.words.txt $OUTDIR/hyp.norm-final.words.txt > $OUTDIR/missing-hyp-ids.list
#python3 ${script_dir}/utils/insert_empty_hyp.py $OUTDIR/missing-hyp-ids.list $OUTDIR/hyp.norm-final.words.txt $OUTDIR/hyp.norm-final.words.withmissing.txt
#python3 ${script_dir}/utils/insert_empty_hyp.py $OUTDIR/missing-hyp-ids.list $OUTDIR/hyp.norm-final.chars.txt $OUTDIR/hyp.norm-final.chars.withmissing.txt
cp $OUTDIR/hyp.norm-final.words.txt $OUTDIR/hyp.norm-final.words.withmissing.txt
cp $OUTDIR/hyp.norm-final.chars.txt $OUTDIR/hyp.norm-final.chars.withmissing.txt

# Step 6. Possible filtering
# TODO
# Currently just cp non-filtered transcripts to filtered transcripts
# This will eventually filter out "bad" uttids that should be removed prior to scoring
cp $OUTDIR/ref.norm-final.words.txt $OUTDIR/ref.norm-final.words.filtered.txt
cp $OUTDIR/ref.norm-final.chars.txt $OUTDIR/ref.norm-final.chars.filtered.txt
cp $OUTDIR/hyp.norm-final.words.withmissing.txt $OUTDIR/hyp.norm-final.words.filtered.txt
cp $OUTDIR/hyp.norm-final.chars.withmissing.txt $OUTDIR/hyp.norm-final.chars.filtered.txt


# Step 7. Now we can run scoring

if [ "$USE_SCLITE" == true ]; then
    # First convert files to uxxxx format
    python3 ${script_dir}/utils/word_trans_utf8_to_uxxxx.py $OUTDIR/ref.norm-final.words.filtered.txt $OUTDIR/ref.norm-final.words.filtered.uxxxx
    python3 ${script_dir}/utils/word_trans_utf8_to_uxxxx.py $OUTDIR/hyp.norm-final.words.filtered.txt $OUTDIR/hyp.norm-final.words.filtered.uxxxx
    python3 ${script_dir}/utils/char_trans_utf8_to_uxxxx.py $OUTDIR/ref.norm-final.chars.filtered.txt $OUTDIR/ref.norm-final.chars.filtered.uxxxx
    python3 ${script_dir}/utils/char_trans_utf8_to_uxxxx.py $OUTDIR/hyp.norm-final.chars.filtered.txt $OUTDIR/hyp.norm-final.chars.filtered.uxxxx

    echo "Computing WER"
    $SCLITE -r $OUTDIR/ref.norm-final.words.filtered.uxxxx -h $OUTDIR/hyp.norm-final.words.filtered.uxxxx -i swb -o all >/dev/null
    wer_sys_file=$OUTDIR/hyp.norm-final.words.filtered.uxxxx.sys

    WER=$(grep 'Sum/Avg' ${wer_sys_file}  | awk '{print $(NF-2)}')
    echo "WER = $WER"

    echo "Computing CER"
    $SCLITE -r $OUTDIR/ref.norm-final.chars.filtered.uxxxx -h $OUTDIR/hyp.norm-final.chars.filtered.uxxxx -i swb -o all >/dev/null
    cer_sys_file=$OUTDIR/hyp.norm-final.chars.filtered.uxxxx.sys

    CER=$(grep 'Sum/Avg' ${cer_sys_file}  | awk '{print $(NF-2)}')
    echo "CER = $CER"

else
    echo "Computing WER"
    LANG=C perl -CSAD $OCR_SCORE --ref_format trn --hyp_format trn $OUTDIR/ref.norm-final.words.filtered.txt $OUTDIR/hyp.norm-final.words.filtered.txt >/dev/null
    wer_sys_file=$OUTDIR/hyp.norm-final.words.filtered.txt.sys

    WER=$(awk '{print $4}' ${wer_sys_file} | head -n 4 | tail -n 1)
    echo "WER = $WER"

    echo "Computing CER"
    LANG=C perl -CSAD $OCR_SCORE --ref_format trn --hyp_format trn $OUTDIR/ref.norm-final.chars.filtered.txt $OUTDIR/hyp.norm-final.chars.filtered.txt >/dev/null
    cer_sys_file=$OUTDIR/hyp.norm-final.chars.filtered.txt.sys

    CER=$(awk '{print $4}' ${cer_sys_file} | head -n 4 | tail -n 1)
    echo "CER = $CER"
fi

num_missing_hyp=$(wc -l $OUTDIR/missing-hyp-ids.list | awk '{print $1}')

echo "Done."
echo ""
echo "For detailed system scores see:"
echo -e "\t${wer_sys_file}"
echo -e "\t${cer_sys_file}"

if [ "$num_missing_hyp" -gt 0 ]; then
echo ""
echo "Warning, you are missing ${num_missing_hyp} hypothesis lines. Your score is penalized due to missing lines."
echo -e "\tFind missing hypothesis ids here: $OUTDIR/missing-hyp-ids.list"
fi
