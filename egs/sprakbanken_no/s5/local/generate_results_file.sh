
#!/usr/bin/env bash

# Copyright 2010-2012 Microsoft Corporation  Johns Hopkins University (Author: Daniel Povey)
# Copyright 2014 Mirsk Digital ApS  (Author: Andreas Kirkedal)
# Copyright 2022  Institute of Language and Speech Processing (ILSP), AthenaRC (Author: Thodoris Kouzelis)
# Apache 2.0.

echo "GMM-based systems" 
cat  exp/*/decode*/scoring_kaldi/best_wer

echo "Chain systems" 
cat  exp/*/*/decode*/scoring_kaldi/best_wer
