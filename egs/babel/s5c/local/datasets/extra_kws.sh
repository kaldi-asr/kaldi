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

function register_extraid {
  local dataset_dir=$1
  local extraid=$2
  echo "Registering $extraid"
  echo $extraid >> $dataset_dir/extra_kws_tasks;
  sort -u $dataset_dir/extra_kws_tasks -o $dataset_dir/extra_kws_tasks
}

function setup_oov_search {
  local phone_cutoff=5

  local g2p_nbest=10
  local g2p_mass=0.95


  local data_dir=$1
  local source_dir=$2
  local extraid=$3

  local kwsdatadir=$data_dir/${extraid}_kws

  mkdir -p $kwsdatadir

  if [ "${dataset_kind}" == "supervised" ] ; then
    for file in $source_dir/rttm ; do
      cp -f $file $kwsdatadir
    done
  fi
  for file in $source_dir/utter_* $source_dir/kwlist*.xml $source_dir/ecf.xml ; do
    cp -f $file $kwsdatadir
  done

  kwlist=$source_dir/kwlist_outvocab.xml
  #Get the KW list
  paste \
    <(cat $kwlist |  grep -o -P "(?<=kwid=\").*(?=\")") \
    <(cat $kwlist | grep -o -P "(?<=<kwtext>).*(?=</kwtext>)" | uconv -f utf-8 -t utf-8 -x Any-Lower) \
    >$kwsdatadir/keywords.txt
  cut -f 2 $kwsdatadir/keywords.txt | \
    perl -ape 's/\s\s*/\n/g;' | sort -u > $kwsdatadir/oov.txt


  #Generate the confusion matrix
  #NB, this has to be done only once, as it is training corpora dependent,
  #instead of search collection dependent
  if [ ! -f exp/conf_matrix/.done ] ; then
    local/generate_confusion_matrix.sh --cmd "$decode_cmd" --nj $my_nj  \
      exp/sgmm5_denlats/dengraph  exp/sgmm5 exp/sgmm5_ali exp/sgmm5_denlats  exp/conf_matrix || return 1
    touch exp/conf_matrix/.done
  fi
  confusion=exp/conf_matrix/confusions.txt

  if [ ! -f exp/g2p/.done ] ; then
    if [ -f data/.extlex ]; then
      local/train_g2p.sh  data/local/lexicon_orig.txt exp/g2p || return 1;
    else
      local/train_g2p.sh  data/local/lexicon.txt exp/g2p || return 1;
    fi
    touch exp/g2p/.done
  fi
  local/apply_g2p.sh --nj $my_nj --cmd "$decode_cmd" \
    --var-counts $g2p_nbest --var-mass $g2p_mass \
    $kwsdatadir/oov.txt exp/g2p $kwsdatadir/g2p || return 1
  L2_lex=$kwsdatadir/g2p/lexicon.lex

  if [ -z "$L1_lex" ] ; then
    L1_lex=data/local/lexiconp.txt
  fi

  local/kws_data_prep_proxy.sh \
    --cmd "$decode_cmd" --nj $my_nj \
    --case-insensitive true \
    --confusion-matrix $confusion \
    --phone-cutoff $phone_cutoff \
    --pron-probs true --beam $proxy_beam --nbest $proxy_nbest \
    --phone-beam $proxy_phone_beam --phone-nbest $proxy_phone_nbest \
    $lang $data_dir $L1_lex $L2_lex $kwsdatadir

}


kws_flags=( --use-icu true )
if [  "${dataset_kind}" == "supervised"  ] ; then
  #The presence of the file had been already verified, so just
  #add the correct switches
  kws_flags+=(--rttm-file $my_rttm_file )
fi
if $my_subset_ecf ; then
  kws_flags+=(--subset-ecf $my_data_list)
fi

if [ ! -f $dataset_dir/.done.kws.oov ] ; then
  setup_oov_search $dataset_dir $dataset_dir/kws oov || exit 1
  register_extraid $dataset_dir oov
  touch $dataset_dir/.done.kws.oov
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
      $my_ecf_file $kwlist $lang ${dataset_dir} || exit 1

    #Register the dataset for default running...
    #We can do it without any problem here -- the kws_stt_tasks will not
    #run it, unless called with --run-extra-tasks true switch
    register_extraid $dataset_dir $extraid
    touch $dataset_dir/.done.kws.$extraid
  done
  for extraid in "${!my_more_kwlists[@]}" ; do
    #The next line will help us in running only one. We don't really
    #know in which directory the KWS setup will reside in, so we will
    #place  the .done file directly into the data directory
    [ -f $dataset_dir/.done.kws.${extraid}_oov ] && continue;
    setup_oov_search $dataset_dir $dataset_dir/${extraid}_kws ${extraid}_oov
    register_extraid $dataset_dir ${extraid}_oov
    touch $dataset_dir/.done.kws.${extraid}_oov
  done
fi

