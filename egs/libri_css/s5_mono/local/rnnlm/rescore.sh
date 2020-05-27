if [ $stage -le 4 ] && $run_lat_rescore; then
  echo "$0: Perform lattice-rescoring on $ac_model_dir"
#  LM=tgsmall # if using the original 3-gram G.fst as old lm
  pruned=
  if $pruned_rescore; then
    pruned=_pruned
  fi
  for decode_set in test_clean test_other dev_clean dev_other; do
    for LM in fglarge tglarge; do 
        decode_dir=${ac_model_dir}/decode_${decode_set}_${LM}
        # Lattice rescoring
        rnnlm/lmrescore$pruned.sh \
            --cmd "$decode_cmd --mem 8G" \
            --weight 0.45 --max-ngram-order $ngram_order \
            data/lang_test_$LM $dir \
            data/${decode_set}_hires ${decode_dir} \
            exp/chain_cleaned/tdnn_1d_sp/decode_${decode_set}_${LM}_${decode_dir_suffix}_rescore
    done
  done
fi

if [ $stage -le 5 ] && $run_nbest_rescore; then
  echo "$0: Perform nbest-rescoring on $ac_model_dir"
  for decode_set in test_clean test_other dev_clean dev_other; do
    for LM in fglarge tglarge; do 
        decode_dir=${ac_model_dir}/decode_${decode_set}_${LM}
        # Nbest rescoring
        rnnlm/lmrescore_nbest.sh \
            --cmd "$decode_cmd --mem 8G" --N 20 \
            0.4 data/lang_test_$LM $dir \
            data/${decode_set}_hires ${decode_dir} \
            exp/chain_cleaned/tdnn_1d_sp/decode_${decode_set}_${LM}_${decode_dir_suffix}_nbest_rescore
    done
  done
fi