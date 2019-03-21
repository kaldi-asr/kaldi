#!/bin/bash

initialize=1
max_speakers=10
statScale=0.2
dubm_model=/export/b02/zili/diarization/callhome/VB/model/diag_ubm_1024_no_delta_final.pkl
ie_model=/export/b02/zili/diarization/callhome/VB/model/extractor_1024_400_no_delta_posterior_scale_0.2_final.pkl

source path.sh

data_dir=$1
init_rttm_filename=$2
output_dir=$3

/home/hzili1/tools/anaconda3/envs/py27/bin/python local/VB_resegmentation.py --initialize $initialize --max-speakers $max_speakers --statScale $statScale --dubm-model $dubm_model --ie-model $ie_model $data_dir/$SGE_TASK_ID $init_rttm_filename $output_dir >> $output_dir/log/VB_resegmentation_$SGE_TASK_ID.log
