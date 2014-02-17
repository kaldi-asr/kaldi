#This script is not really supposed to be run directly 
#Instead, it should be sourced from the decoding script
#It makes many assumption on existence of certain environmental
#variables as well as certain directory structure.

if [ "${dataset_kind}" == "supervised" ] ; then
  mandatory_variables="my_ecf_file my_kwlist_file my_rttm_file" 
  optional_variables="my_subset_ecf"
else
  mandatory_variables="my_ecf_file my_kwlist_file" 
  optional_variables="my_subset_ecf"
fi

check_variables_are_set

if [ "$dataset_kind" == "shadow" ]; then
  true #we do not support multiple kw lists for shadow set system
   
else # This will work for both supervised and unsupervised dataset kinds
  kws_flags=()
  if [  "${dataset_kind}" == "supervised"  ] ; then
    #The presence of the file had been already verified, so just 
    #add the correct switches
    kws_flags+=(--rttm-file $my_rttm_file )
  fi
  if $my_subset_ecf ; then
    kws_flags+=(--subset-ecf $my_data_list)
  fi
  
  if [ ${#my_more_kwlists[@]} -ne 0  ] ; then
    
    touch $dataset_dir/extra_kws_tasks
    
    for extraid in "${!my_more_kwlists[@]}" ; do
      #The next line will help us in running only one. We don't really
      #know in which directory the KWS setup will reside in, so we will 
      #place  the .done file directly into the data directory
      [ -f $dataset_dir/.done.kws.$extraid ] && continue;
      kwlist=${my_more_kwlists[$extraid]}

      local/kws_setup.sh  --extraid $extraid --case_insensitive $case_insensitive \
        "${kws_flags[@]}" "${icu_opt[@]}" \
        $my_ecf_file $kwlist data/lang ${dataset_dir} || exit 1
     
      #Register the dataset for default running...
      #We can do it without any problem here -- the kws_stt_tasks will not
      #run it, unless called with --run-extra-tasks true switch
      echo $extraid >> $dataset_dir/extra_kws_tasks;  
      sort -u $dataset_dir/extra_kws_tasks -o $dataset_dir/extra_kws_tasks
      
      touch $dataset_dir/.done.kws.$extraid
    done
  fi
fi

