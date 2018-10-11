#!/bin/bash

# "Our baselines"
echo "########################################"
echo "GMM RESULTs:"
dt_real_2ch_beamformit dt_simu_2ch_beamformit et_real_2ch_beamformit et_simu_2ch_beamformit dt_real_8ch_beamformit dt_simu_8ch_beamformit et_real_8ch_beamformit et_simu_8ch_beamformit
echo "exp/tri3/decode_dt_real_1ch"
cat exp/tri3/decode_dt_real_1ch/scoring_kaldi/best_wer*
echo ""
echo "exp/tri3/decode_dt_simu_1ch"
cat exp/tri3/decode_dt_simu_1ch/scoring_kaldi/best_wer*
echo ""
echo "exp/tri3/decode_et_real_1ch"
cat exp/tri3/decode_et_real_1ch/scoring_kaldi/best_wer*
echo ""
echo "exp/tri3/decode_et_simu_1ch"
cat exp/tri3/decode_et_simu_1ch/scoring_kaldi/best_wer*
echo ""
echo "exp/tri3/decode_dt_real_2ch_beamformit"
cat exp/tri3/decode_dt_real_2ch_beamformit/scoring_kaldi/best_wer*
echo ""
echo "exp/tri3/decode_dt_simu_2ch_beamformit"
cat exp/tri3/decode_dt_simu_2ch_beamformit/scoring_kaldi/best_wer*
echo ""
echo "exp/tri3/decode_et_real_2ch_beamformit"
cat exp/tri3/decode_et_real_2ch_beamformit/scoring_kaldi/best_wer*
echo ""
echo "exp/tri3/decode_et_simu_2ch_beamformit"
cat exp/tri3/decode_et_simu_2ch_beamformit/scoring_kaldi/best_wer*
echo ""
echo "exp/tri3/decode_dt_real_8ch_beamformit"
cat exp/tri3/decode_dt_real_8ch_beamformit/scoring_kaldi/best_wer*
echo ""
echo "exp/tri3/decode_dt_simu_8ch_beamformit"
cat exp/tri3/decode_dt_simu_8ch_beamformit/scoring_kaldi/best_wer*
echo ""
echo "exp/tri3/decode_et_real_8ch_beamformit"
cat exp/tri3/decode_et_real_8ch_beamformit/scoring_kaldi/best_wer*
echo ""
echo "exp/tri3/decode_et_simu_8ch_beamformit"
cat exp/tri3/decode_et_simu_8ch_beamformit/scoring_kaldi/best_wer*
echo "########################################"
echo "TDNN RESULTs:"
echo "exp/chain_tr_simu_8ch/tdnn1a_sp/decode_test_tg_5k_dt*"
cat exp/chain_tr_simu_8ch/tdnn1a_sp/decode_test_tg_5k_dt*/scoring_kaldi/best_wer_*
echo ""
echo "exp/chain_tr_simu_8ch/tdnn1a_sp/decode_test_tg_5k_et*"
cat exp/chain_tr_simu_8ch/tdnn1a_sp/decode_test_tg_5k_et*/scoring_kaldi/best_wer_*
