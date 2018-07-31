#This script is not really supposed to be run directly
#Instead, it should be sourced from the decoding script
#It makes many assumption on existence of certain environmental
#variables as well as certain directory structure.

if [ "${dataset_kind}" == "supervised" ] ; then
  mandatory_variables="my_ecf_file my_kwlists my_rttm_file"
  optional_variables="my_subset_ecf"
else
  mandatory_variables="my_ecf_file my_kwlists"
  optional_variables="my_subset_ecf"
fi

check_variables_are_set

if [ ! -f ${dataset_dir}/kws/.done ] ; then
  kws_flags=( --use-icu true )
  if [  "${dataset_kind}" == "supervised"  ] || [ !-z "$my_rttm_file" ] ; then
    kws_flags+=(--rttm-file $my_rttm_file )
  fi
  if $my_subset_ecf ; then
    kws_flags+=(--subset-ecf $my_data_list)
  fi
  local/kws_setup.sh --case_insensitive $case_insensitive \
    "${kws_flags[@]}" "${icu_opt[@]}" \
    $my_ecf_file $my_kwlist_file $lang ${dataset_dir} || exit 1
  touch ${dataset_dir}/kws/.done
fi
