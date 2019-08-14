
#!/bin/bash


steps/scoring/score_kaldi_wer.sh "$@"
steps/scoring/score_kaldi_cer.sh --stage 2 "$@"
