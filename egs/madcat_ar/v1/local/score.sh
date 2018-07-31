#!/bin/bash


steps/scoring/score_kaldi_wer.sh --word_ins_penalty 0.0,0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0,6.5,7.0 "$@"
steps/scoring/score_kaldi_cer.sh --stage 2 --word_ins_penalty 0.0,0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0,6.5,7.0 "$@"
