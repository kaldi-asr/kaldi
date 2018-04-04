case "$USER" in
"ac1nmx")
  # CHiME Challenge wav root (after unzipping)...
  export WAV_ROOT="/data/ac1nmx/data/PCCdata16kHz" 

  # Used by the recogniser for storing data/ exp/ mfcc/ etc
  export REC_ROOT="." 
  ;;
*)
  echo "Please define WAV_ROOT and REC_ROOT for user $USER"
  ;;
esac

