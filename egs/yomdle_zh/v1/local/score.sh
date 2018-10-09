#!/bin/bash


steps/scoring/score_kaldi_wer.sh --max-lmwt 10 "$@"
steps/scoring/score_kaldi_cer.sh --max-lmwt 10 --stage 2 "$@"
