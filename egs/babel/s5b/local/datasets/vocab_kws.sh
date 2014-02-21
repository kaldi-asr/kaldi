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
   
elif [ ! -f $dataset_dir/.done.kws.fullvocab ] ; then
  #a This will work for both supervised and unsupervised dataset kinds
  kws_flags=()
  if [ "$dataset_kind" == "supervised" ] ; then
    kws_flags+=(--rttm-file $my_rttm_file )
  fi
  if $my_subset_ecf ; then
    kws_flags+=(--subset-ecf $my_data_list)
  fi
  
  #We just could come with some bogus naming scheme,
  #but as long as the audio files can tell the iarpa lang id, we will use that
  langid=`ls -1 $my_data_dir/audio/ | head -n 1| cut -d '_' -f 3`

  #NB: we assume the default KWS search is already done and will "borrow" 
  #the rttm and ecf files.
  #We could easily generate the ecf file, but the RTTM assumes the decoding
  #had been already done. That could be done 
  #Ideally, these files should be generated here!

  local/kws_setup.sh --kwlist-wordlist true "${kws_flags[@]}"  \
    --extraid fullvocab $my_ecf_file \
      <(cat data/lang/words.txt | \
        grep -v -F "<" | grep -v -F "#"  | \
        awk "{printf \"KWID$langid-FULLVOCAB-%05d %s\\n\", \$2, \$1 }" ) \
    data/lang ${dataset_dir} || exit 1

  echo fullvocab >> $dataset_dir/extra_kws_tasks;  
  sort -u $dataset_dir/extra_kws_tasks -o  $dataset_dir/extra_kws_tasks
  touch $dataset_dir/.done.kws.fullvocab
fi


