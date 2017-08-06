
echo "GMM-based systems" 
for x in exp/*/decode*;do
    [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh;
done 

echo "nnet3 xent systems" 
for x in exp/nnet3/tdnn*/decode* exp/nnet3/lstm*/decode* ;do
    [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh;
done 

echo "Nnet3 chain systems" 
for x in exp/chain/tdnn*/decode* exp/chain/lstm*/decode*;do
    [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh;
done

