if [[ $(hostname -f) == *.clsp.jhu.edu ]] ; then
  REVERB_home=/export/corpora5/REVERB_2014/REVERB
  export wsjcam0=/export/corpora3/LDC/LDC95S24/wsjcam0
  # set LDC WSJ0 directory to obtain LMs 
  # REVERB data directory only provides bi-gram (bcb05cnp), but this recipe also uses 3-gram (tcb05cnp.z)
  export wsj0=/export/corpora5/LDC/LDC93S6A/11-13.1 #LDC93S6A or LDC93S6B
  # It is assumed that there will be a 'wsj0' subdirectory
  # within the top-level corpus directory
else
  echo "Set the data directory locations." && exit 1;
fi

export reverb_dt=$REVERB_home/REVERB_WSJCAM0_dt
export reverb_et=$REVERB_home/REVERB_WSJCAM0_et
export reverb_real_dt=$REVERB_home/MC_WSJ_AV_Dev
export reverb_real_et=$REVERB_home/MC_WSJ_AV_Eval

