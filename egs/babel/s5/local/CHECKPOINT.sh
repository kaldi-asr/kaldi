#!/bin/bash

function GETAPPROVAL {
  until false ; do 
    echo "Do you want to run the command (y/n)?"
    read -n 1 WISH
    
    if [ "$WISH" == "y" ]; then 
      return true;
    elif [ "$WISH" == "n" ]; then
      return false;
    fi

  done
}

function ESCAPE_PARAMS {
  local out=""

  for v in "$@"; do

      if [[ "$v" == *"<"* ]]; then
         out="$out \"$v\""
      elif [[ "$v" == *">"* ]] ; then 
         out="$out \"$v\""
      elif [[ "$v" == *"|"* ]] ; then 
         out="$out \'$v\'"
      elif [[ "$v" == *" "* ]] ; then 
         out="$out \"$v\""
      else
         out="$out $v"
      fi
  done

  echo $out
}

function CHK {
  local ID=DEFAULT
  CHECKPOINT $ID "$@"
}

function CHECKPOINT {
  COLOR_GREEN='\e[00;32m'
  COLOR_RED='\e[00;31m'
  COLOR_BLUE='\e[00;34m'
  COLOR_DEFAULT='\e[00m'

  local ID=$1; shift
  #We retrieve the counter variable we use for checkpointing
  #Because the name of the variable is govern by the checkpoint ID
  #we must use indirect approach
  local COUNTER_NAME="CHECKPOINT_${ID}_COUNTER"
  local COUNTER
  eval COUNTER=\$$COUNTER_NAME
  if [ -z $COUNTER ]; then
    COUNTER=0
  fi
  echo -e  ${COLOR_GREEN}CHECKPOINT:$ID, COUNTER=$COUNTER $COLOR_DEFAULT >&2

  #Now the same for "LAST GOOD STATE"
  if [ "$ID" == "DEFAULT" ]; then
    local LAST_GOOD_NAME="LAST_GOOD"
  else
    local LAST_GOOD_NAME="LAST_GOOD_$ID"
  fi
  local LAST_GOOD_VALUE
  eval LAST_GOOD_VALUE=\$$LAST_GOOD_NAME

  echo -e ${COLOR_GREEN}"CHECKPOINT: $LAST_GOOD_NAME=$LAST_GOOD_VALUE"${COLOR_DEFAULT} >&2

  #The command has to be run, if no-checkpoint tracking is in progress
  #or we are already gone through the last problematic part
  if [ -z $LAST_GOOD_VALUE  ] || [ $COUNTER -ge $LAST_GOOD_VALUE ]; then
    #bash print_args.sh `ESCAPE_PARAMS $CMD`

    if [ !$INTERACTIVE_CHECKPOINT ] ; then
      eval `ESCAPE_PARAMS "$@"`
    else 
      APPROVAL=GETAPPROVAL
      if $APPROVAL ; then
        eval `ESCAPE_PARAMS $@`
      fi
    fi

    if [ $? -ne 0 ] ; then
      echo -e ${COLOR_RED}"CHECKPOINT FAILURE: The command returned non-zero status" >&2
      echo -e "                    rerun the script with the parameter -c $LAST_GOOD_NAME=$COUNTER" >&2
      echo -e "COMMAND">&2
      echo -e "  " "$@" ${COLOR_RED} >&2 

      exit 1
    fi
  else
    #Else, we just skip the command....
    echo -e ${COLOR_GREEN}"CHECKPOINT: SKIPPING, $LAST_GOOD_NAME=$COUNTER" >&2
    echo -e "$@"${COLOR_DEFAULT} >&2
  fi

  COUNTER=$(( $COUNTER + 1 )) 
  eval export $COUNTER_NAME=$COUNTER
}

while getopts ":c:i" opt; do
  case $opt in
    c)
      eval $OPTARG
      ;;
    i)
      INTERACTIVE_CHECKPOINT=true
  esac
done


