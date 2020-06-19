#!/usr/bin/env bash

# DESC: Evaluate text detection bounding boxes
#
# INPUT:
#    truth - Directory of input truth csv files
#        formt: ID,name,col1,row1,col2,row2,col3,row3,col4,row4,box_conf
#    predict - Directory of input predict csv files
#        formt: ID,name,col1,row1,col2,row2,col3,row3,col4,row4,box_conf
#    iou - intersection over union
#    
# OUTPUT:
#    output - output directory of results (plot)
#    log - log output

source activate py35
echo "LD_LIBRARY_PATH = ${LD_LIBRARY_PATH}"
echo "PATH = ${PATH}"

LANGUAGE=Farsi
IOU=0.50
TRUTH_CSV=/exp/scale18/ocr/data/derived/SLAM_2.0/${LANGUAGE}/transcribed/truth_csv
PREDICT_CSV=/exp/detter/scale18/slam2/results/${LANGUAGE}/transcribed/bbox_csv
RESULTS=/exp/detter/scale18/slam2/results/${LANGUAGE}/transcribed/bbox_results
LOG=/exp/detter/scale18/slam2/results/${LANGUAGE}/transcribed/bbox_results_log.txt

echo "Language ${LANGUAGE}"
echo "Truth ${TRUTH_CSV}"
echo "Predict ${PREDICT_CSV}"
echo "Results ${RESULTS}"

echo "\n...evalulate text detection"
python /exp/detter/scale18/ocr/cv_scale/eval/eval_detect_lines.py \
--truth=${TRUTH_CSV} \
--predict=${PREDICT_CSV} \
--iou=${IOU} \
--output=${RESULTS} \
--log=${LOG}

echo "...COMPLETE..."
