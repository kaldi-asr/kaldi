./local/multi-stream/run-dnn-mstrm-bnfeats_train.sh --stage 0

./local/multi-stream/run-get-CombX_bnfeats.sh --njdec 112 --comb-num 511 --stage 0
./local/multi-stream/run-dnn-mstrm-bnfeats_test.sh --njdec 112 --test-bn data-fbank-bn-fbank-traps_mstrm_9strms-2BarkPerStrm_CMN_bnfeats_splice5_traps_dct_basis6_iters-per-epoch5/test_eval92_percond-spk_strm-mask_Comb511


./local/multi-stream/run-get-autoencoder_mdeltaPM_bnfeats.sh --njdec 112 --test data-multistream-fbank/test_eval92_percond-spk --stage 0
./local/multi-stream/run-dnn-mstrm-bnfeats_test.sh --njdec 112 --test-bn data-fbank-bn-fbank-traps_mstrm_9strms-2BarkPerStrm_CMN_bnfeats_splice5_traps_dct_basis6_iters-per-epoch5/test_eval92_percond-spk_strm-mask_autoencoder-mdeltaPM

