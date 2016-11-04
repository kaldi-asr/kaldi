#!/bin/bash
#
# Copyright  2014 Nickolay V. Shmyrev 
#            2014 Brno University of Technology (Author: Karel Vesely)
# Apache 2.0

# To be run from one directory above this script.

. path.sh

export LC_ALL=C

# Prepare: test set 2014 iwslt
for set in tst2014; do
  dir=data/$set
  mkdir -p $dir

  # Merge transcripts into a single 'stm' file, do some mappings:
  # - <F0_M> -> <o,f0,male> : map dev stm labels to be coherent with train + test,
  # - <F0_F> -> <o,f0,female> : --||--
  # - (2) -> null : remove pronunciation variants in transcripts, keep in dictionary
  # - <sil> -> null : remove marked <sil>, it is modelled implicitly (in kaldi)
  # - (...) -> null : remove utterance names from end-lines of train
  # - it 's -> it's : merge words that contain apostrophe (if compound in dictionary, local/join_suffix.py)
  { # Add STM header, so sclite can prepare the '.lur' file
    echo ';;
;; LABEL "o" "Overall" "Overall results"
;; LABEL "f0" "f0" "Wideband channel"
;; LABEL "f2" "f2" "Telephone channel"
;; LABEL "male" "Male" "Male Talkers"
;; LABEL "female" "Female" "Female Talkers"
;; LABEL "unknown" "unknown" "Unknown Talkers"
;;'
    # Process the STMs
    cat db/TEDLIUM_release2/$set/*.stm | sort -k1,1 -k2,2 -k4,4n | \
      sed -e 's:<F0_M>:<o,f0,male>:' \
          -e 's:<F0_F>:<o,f0,female>:' \
          -e 's:([0-9])::g' \
          -e 's:<sil>::g' \
          -e 's:([^ ]*)$::' | \
      awk '{ $2 = "A"; print $0; }'
  } | local/join_suffix.py db/cantab-TEDLIUM/cantab-TEDLIUM.dct > data/$set/stm 

  # Prepare 'text' file
  # - {NOISE} -> [NOISE] : map the tags to match symbols in dictionary
  cat $dir/stm | grep -v -e 'ignore_time_segment_in_scoring' -e ';;' | \
    awk '{ printf ("%s-%07d-%07d", $1, $4*100, $5*100); 
           for (i=7;i<=NF;i++) { printf(" %s", $i); } 
           printf("\n"); 
         }' | tr '{}' '[]' | sort -k1,1 > $dir/text || exit 1

  # Prepare 'segments', 'utt2spk', 'spk2utt'
  cat $dir/text | cut -d" " -f 1 | awk -F"-" '{printf("%s %s-%s %07.2f %07.2f\n", $0, $1, $2, $3/100.0, $4/100.0)}' > $dir/segments
  cat $dir/segments | awk '{print $1, $2}' > $dir/utt2spk
  cat $dir/utt2spk | utils/utt2spk_to_spk2utt.pl > $dir/spk2utt
  
  # Prepare 'wav.scp', 'reco2file_and_channel' 
  cat $dir/spk2utt | cut -d " " -f1 | cut -d "." -f 2- | awk -v set=$set -v pwd=$PWD '{ printf("ted.%s sph2pipe -f wav -p %s/db/TEDLIUM_release2/%s/sph/IWSLT14.ASR.%s.sph |\n", $1, pwd, set, $1); }' > $dir/wav.scp
  cat $dir/wav.scp | awk '{ print $1, $1, "A"; }' > $dir/reco2file_and_channel
  
  # Create empty 'glm' file
  echo ';; empty.glm
  [FAKE]     =>  %HESITATION     / [ ] __ [ ] ;; hesitation token
  ' > data/$set/glm

  # Check that data dirs are okay!
  utils/validate_data_dir.sh --no-feats $dir || exit 1
done

