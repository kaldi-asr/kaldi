#This script is not really supposed to be run directly 
#Instead, it should be sourced from the decoding script
#It makes many assumption on existence of certain environmental
#variables as well as certain directory structure.

eval my_data_cmudb=\$${dataset_type}_data_cmudb

if [ "${dataset_kind}" != "supervised" ] ; then
  mandatory_variables="my_data_dir my_data_list my_nj my_data_cmudb" 
  optional_variables=""
else
  mandatory_variables="my_data_dir my_data_list my_nj my_data_cmudb"
  optional_variables="my_stm_file"
fi

check_variables_are_set

if [[ ! -f ${dataset_dir}/wav.scp || ${dataset_dir}/wav.scp -ot "$my_data_cmudb" ]]; then
  echo ---------------------------------------------------------------------
  echo "Preparing ${dataset_type} data lists in ${dataset_dir} on" `date`
  echo ---------------------------------------------------------------------
  mkdir -p ${dataset_dir}
  local/cmu_uem2kaldi_dir.sh --filelist $my_data_list \
    $my_data_cmudb  $my_data_dir ${dataset_dir}
fi

if [ "$dataset_kind" == "supervised" ]; then
  echo ---------------------------------------------------------------------
  echo "Preparing ${dataset_type} stm files in ${dataset_dir} on" `date`
  echo ---------------------------------------------------------------------
  if [ ! -z $my_stm_file ] ; then
    local/augment_original_stm.pl $my_stm_file ${dataset_dir}
  else
    local/prepare_stm.pl --fragmentMarkers \-\*\~ ${dataset_dir}
  fi
fi
