#!/bin/bash

export NJ=`(. ./lang.conf > /dev/null; echo $train_nj )`
export TYPE=`(. ./lang.conf > /dev/null; echo $babel_type )`

echo $NJ
echo $TYPE

if [ "$TYPE" == "limited" ]; then
  T_SHORT="6:0:0"
  T_MEDIUM="12:0:0"
  T_LONG="24:0:0"
  T_EXTREME="48:0:0"
  BNF_NJ=$((16 * 4))
  DNN_NJ=$((16 * 4))
elif [ "$TYPE" == "full" ]; then
  T_SHORT="6:0:0"
  T_MEDIUM="24:0:0"
  T_LONG="48:0:0"
  T_EXTREME="48:0:0"
  BNF_NJ=$((16 * 8))
  DNN_NJ=$((16 * 8))
else
  echo "Unknown BABEL type! Exiting..."
  exit 1
fi


export SBATCH_JOBID


function sbatch {
  #echo "sbatch " "${@}"
  output_name=""
  for param in "${@}"; do
    if [[ $param =~ ^\./.*sh ]]; then
      output_name=`basename $param`
    fi
  done
  if [ ! -z $output_name ]; then
    output_name="-o ${output_name}.%j"
  fi
  #echo "OUTPUT: $output_name"
  echo /usr/bin/sbatch --mail-type ALL --mail-user 'jtrmal@gmail.com' $output_name "${@}" 
  jobid=$(/usr/bin/sbatch --mail-type ALL --mail-user 'jtrmal@gmail.com' $output_name "${@}" | tee /dev/stderr  | grep "Submitted batch job" | awk '{print $4}'  )
  SBATCH_JOBID=$jobid
}

sbatch -p normal -n $NJ -t $T_SHORT  ./run-1-main.sh --tri5-only true 
TRI5_ID=$SBATCH_JOBID

sbatch -p normal -n $NJ -t $T_LONG --dependency=afterok:$TRI5_ID  ./run-1-main.sh 
PLP_ID=$SBATCH_JOBID

sbatch -p normal -n $NJ -t $T_SHORT --dependency=afterok:$TRI5_ID ./run-2-segmentation.sh
SEG_ID=$SBATCH_JOBID

if [ "$TYPE" == "limited" ]; then
  sbatch -p gpu -n $DNN_NJ -t $T_MEDIUM --dependency=afterok:$TRI5_ID  ./run-2a-nnet-ensemble-gpu.sh --dir exp/tri6_nnet/
else
  sbatch -p gpu -n $DNN_NJ -t $T_MEDIUM --dependency=afterok:$TRI5_ID  ./run-2a-nnet-gpu.sh
  DNN_ID=$SBATCH_JOBID
  sbatch -p gpu -n $DNN_NJ -t $T_MEDIUM --dependency=afterok:$DNN_ID  ./run-2a-nnet-mpe.sh
fi
DNN_ID=$SBATCH_JOBID

sbatch -p gpu -n $BNF_NJ -t 24:0:0 --dependency=afterok:$TRI5_ID  ./run-8a-kaldi-bnf.sh
BNF_ID=$SBATCH_JOBID

sbatch -p normal -n $NJ   -t $T_LONG --dependency=afterok:$BNF_ID  ./run-8b-kaldi-bnf-sgmm.sh
BNF_SGMM_ID=$SBATCH_JOBID


#Decode DNNs and PLP systems
sbatch -p normal -n 128 -t $T_MEDIUM  --dependency=afterok:$DNN_ID:$PLP_ID ./run-5-anydecode.sh --fast-path true --skip-kws true  --type dev10h
DECODE_DNN_PLP_ID=$SBATCH_JOBID
sbatch -p normal -n 16 -t $T_MEDIUM  --dependency=afterok:$DECODE_DNN_PLP_ID ./run-5-anydecode.sh --fast-path true

#Decode BNF systems
sbatch -p normal -n 128  -t $T_LONG --dependency=afterok:$BNF_SGMM_ID:$DECODE_DNN_PLP_ID ./run-8d-test-kaldi-bnf-sgmm.sh --skip-kws true --type dev10h
DECODE_BNF_SGMM_ID=$SBATCH_JOBID
sbatch -p normal -n 16  -t $T_MEDIUM --dependency=afterok:$DECODE_BNF_SGMM_ID ./run-8d-test-kaldi-bnf-sgmm.sh

exit 0

#For the discriminative training, we have to actually decode the unsup.seg
#The unsup.seg needs segmentation to be done, i.e. it depends on the individual systems and on the segmentation
if [ "$TYPE" == "limited" ]; then
  #First, setup data
  sbatch -p normal -n $NJ -t $T_LONG --dependency=afterok:$SEG_ID ./run-4-anydecode.sh --fast-path true --skip-scoring true --skip-kws true --dir unsup.seg --data-only true
  UNSUP_DATA_PREPARED=$SBATCH_JOBID
  sbatch -p normal -n 256 -t $T_LONG --dependency=afterok:$UNSUP_DATA_PREPARED:$DNN_ID:$PLP_ID ./run-4-anydecode.sh --fast-path true --skip-scoring true --skip-kws true --dir unsup.seg
  SEMI_PARTA_ID=$SBATCH_JOBID
  sbatch -p normal -n 256 -t $T_LONG --dependency=afterok:$UNSUP_DATA_PREPARED:$BNF_SGMM_ID:$DECODE_DNN_PLP_ID ./run-8d-test-kaldi-bnf-sgmm.sh --skip-kws true --skip-kws true --type unsup.seg 
  SEMI_PARTB_ID=$SBATCH_JOBID
fi

#
#
#We do not run BNF on the top of DNN by default (low performance)
#sbatch -p gpu -n $BNF_NJ -t 24:0:0 --dependency=afterok:$BNF_ID  ./run-8c-kaldi-bnf-dnn.sh
#BNF_DNN_ID=$SBATCH_JOBID
#The decoding depends on the BNF-SGMM in that sense that it expects the data directories to be prepared.
#It can create the directories on its own, but do not run those two scripts in parallel -- because of no locking
#this will result in crash as the scripts will overwrite each others's files
#sbatch -p normal -n 128  -t $T_LONG --dependency=afterok:$BNF_DNN_ID:$DECODE_DNN_PLP_ID:$DECODE_BNF_SGMM_ID ./run-8e-test-kaldi-bnf-dnn.sh --skip-kws true
#DECODE_BNF_DNN_ID=$SBATCH_JOBID
#sbatch -p normal -n 16 -t $T_MEDIUM --dependency=afterok:$DECODE_BNF_DNN_ID ./run-8e-test-kaldi-bnf-dnn.sh


