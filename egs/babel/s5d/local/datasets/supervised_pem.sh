#This script is not really supposed to be run directly
#Instead, it should be sourced from the decoding script
#It makes many assumption on existence of certain environmental
#variables as well as certain directory structure.
if [ "${dataset_type}" != "supervised" ] ; then
  mandatory_variables="my_data_dir my_data_list my_nj "
  optional_variables=""
else
  mandatory_variables="my_data_dir my_data_list my_nj "
  optional_variables="my_stm_file "
fi

check_variables_are_set


if [[ ! -f ${dataset_dir}/wav.scp || ${dataset_dir}/wav.scp -ot "$my_data_dir" ]]; then
  echo ---------------------------------------------------------------------
  echo "Preparing ${dataset_type} data lists in ${dataset_dir} on" `date`
  echo ---------------------------------------------------------------------
  mkdir -p ${dataset_dir}
  local/prepare_acoustic_training_data.pl --fragmentMarkers \-\*\~  \
    $my_data_dir ${dataset_dir} > ${dataset_dir}/skipped_utts.log || exit 1
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

