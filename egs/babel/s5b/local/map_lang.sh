#! /usr/bin/bash

VARIABLES=`diff <(compgen -A variable) <(. ./lang.conf.orig; compgen -A variable) | grep '^>'| sed 's/^> *//g'`

. ./conf/common_vars.sh
. ./lang.conf.orig

for variable in $VARIABLES ; do

    eval VAL=\$${variable}
    if [[ $VAL =~ /export/babel/data/ ]] ; then
      eval $variable=${VAL/${BASH_REMATCH[0]}/"/work/02359/jtrmal/"/}
      #declare -x $variable
      declare -p $variable
    fi
done

for kwlist in $( (compgen -A variable) | grep _data_list ) ; do
  declare -p $kwlist
  eval KEYS="\${!${kwlist}[@]}"
  #declare -p my_more_kwlist_keys
  for key in $KEYS  # make sure you include the quotes there
  do
    #echo $key
    eval VAL="\${${kwlist}[$key]}"
    #echo $my_more_kwlist_val
    if [[ $VAL =~ /export/babel/data/ ]] ; then
      eval $kwlist["$key"]=${VAL/${BASH_REMATCH[0]}/"/work/02359/jtrmal/"/}
    fi
  done
  declare -p $kwlist
done
unset VAL
unset KEYS

for kwlist in $( (compgen -A variable) | grep _data_dir ) ; do
  declare -p $kwlist
  eval KEYS="\${!${kwlist}[@]}"
  #declare -p my_more_kwlist_keys
  for key in $KEYS  # make sure you include the quotes there
  do
    #echo $key
    eval VAL="\${${kwlist}[$key]}"
    #echo $my_more_kwlist_val
    if [[ $VAL =~ /export/babel/data/ ]] ; then
      eval $kwlist["$key"]=${VAL/${BASH_REMATCH[0]}/"/work/02359/jtrmal/"/}
    fi
  done
  declare -p $kwlist
done
unset VAL
unset KEYS

for kwlist in $( (compgen -A variable) | grep _more_kwlists ) ; do
  declare -p $kwlist
  eval KEYS="\${!${kwlist}[@]}"
  #declare -p my_more_kwlist_keys
  for key in $KEYS  # make sure you include the quotes there
  do
    #echo $key
    eval VAL="\${${kwlist}[$key]}"
    #echo $my_more_kwlist_val
    if [[ $VAL =~ /export/babel/data/ ]] ; then
      eval $kwlist["$key"]=${VAL/${BASH_REMATCH[0]}/"/work/02359/jtrmal/"/}
    fi
  done
  declare -p $kwlist
done
unset VAL
unset KEYS

if [ "$babel_type" == "limited" ] ; then
  train_nj=32
else
  train_nj=64
fi
dev10h_nj=60
unsup_nj=120
shadow_nj=60
shadow2_nj=120
eval_nj=120
