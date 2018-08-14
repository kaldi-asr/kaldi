#!/bin/bash

#
#
#  SCALE 2018 Text detection Baseline with Tesseract 4
#
#  last updated: 2018-05-14
#
#  INPUT:
#		LANGUAGE - SLAM language to evaluate
#		TRUTH_CSV - Transcription annotation csv file 
#			formt: ID,name,col1,row1,col2,row2,col3,row3,col4,row4,confidence,truth,rotation,quality,script
#
#		PREDICT_CSV - The predicted transcription csv file
#			formt: ID,name,col1,row1,col2,row2,col3,row3,col4,row4,confidence,truth
#
#  OUTPUT:
#		OUTPUT_DIR - 
#
#
#  qsub -v PATH -S /bin/bash -b y -q all.q -cwd -j y -N detter -l num_proc=4,mem_free=64G /exp/detter/scale18/ocr/cv_scale/scripts/baseline_text_detect.sh
#
#  Original data - /exp/scale18/ocr/data/SLAM_2.0/FINAL_SLAM_sanitized
#  Derived data (truth_csv, truth_overlay, ...) - /exp/scale18/ocr/data/derived/SLAM_2.0
#  Tools (Tesseract, Leptonica, Tesserocr) - /exp/scale18/ocr/tools
#

source activate py35
export LD_LIBRARY_PATH=/exp/scale18/ocr/tools/leptonica-1.74.4/lib:/exp/scale18/ocr/tools/tesseract/install/lib:$LD_LIBRARY_PATH

echo "LD_LIBRARY_PATH = ${LD_LIBRARY_PATH}"
echo "PATH = ${PATH}"

MODELS_DIR=/exp/scale18/ocr/tools/tessdata
MODELS_LANG=far+eng
LANGUAGE=Farsi
INPUT=/exp/scale18/ocr/data/derived/SLAM_2.0/${LANGUAGE}/transcribed_list.txt
OUTPUT=/exp/detter/scale18/slam2/results/${LANGUAGE}/transcribed/bbox_csv
OVERLAY=/exp/detter/scale18/slam2/results/${LANGUAGE}/transcribed/bbox_overlay
LOG=/exp/detter/scale18/slam2/results/${LANGUAGE}/transcribed/bbox_log.txt

echo "Models ${MODELS_DIR}"
echo "Model lang ${MODELS_LANG}"

echo "Language ${LANGUAGE}"
echo "Input ${INPUT}"
echo "Output ${OUTPUT}"
echo "Overlay ${OVERLAY}"

echo "...evalulate text detection"
python /exp/detter/scale18/ocr/cv_scale/detect_lines/get_bbox_tesserocr.py \
--tess_data=${MODELS_DIR} \
--lang=${MODELS_LANG} \
--oem=1 \
--blur=0 \
--line=1 \
--input=${INPUT} \
--output=${OUTPUT} \
--overlay=${OVERLAY} \
--log=${LOG}

echo "...COMPLETE..."
