## Introduction
This recips is based ont the "NER-Trs-Vol1" corpus (selected from National Education Radio archive, which could be applied for "Non-Commenrical Use Only". For more details, please visit [Formosa Speech in the Wild (FSW)](https://sites.google.com/speech.ntut.edu.tw/fsw)
project.

*  Before you run this recips, please apply, download and put or make a link of the corpus under this folder.

*  Then, you excute "run.sh" to train models and decode test data. There are many switchs in this sceript (flag "true" or "false"). you could set them to "false" to bypass centain steps.
), for example:    

    true && (
        local/prepare_dict.sh || exit 1;    
    ).   

*  Finally, run "result.sh" to collect all decoding results. 

## Results:

WER: 
* %WER 61.32 [ 83373 / 135972, 5458 ins, 19156 del, 58759 sub ] exp/mono/decode_test/wer_11_0.0
* %WER 41.00 [ 55742 / 135972, 6725 ins, 12763 del, 36254 sub ] exp/tri1/decode_test/wer_15_0.0
* %WER 40.41 [ 54948 / 135972, 7366 ins, 11505 del, 36077 sub ] exp/tri2/decode_test/wer_14_0.0
* %WER 38.67 [ 52574 / 135972, 6855 ins, 11250 del, 34469 sub ] exp/tri3a/decode_test/wer_15_0.0
* %WER 35.70 [ 48546 / 135972, 7197 ins, 9717 del, 31632 sub ] exp/tri4a/decode_test/wer_17_0.0
* %WER 39.70 [ 53982 / 135972, 7199 ins, 11014 del, 35769 sub ] exp/tri4a/decode_test.si/wer_15_0.0
* %WER 32.11 [ 43661 / 135972, 6112 ins, 10185 del, 27364 sub ] exp/tri5a/decode_test/wer_17_0.5
* %WER 35.93 [ 48849 / 135972, 6611 ins, 10427 del, 31811 sub ] exp/tri5a/decode_test.si/wer_13_0.5
* %WER 24.43 [ 33218 / 135972, 5524 ins, 7583 del, 20111 sub ] exp/nnet3/tdnn_sp/decode_test/wer_12_0.0


CER: 
* %WER 54.09 [ 116688 / 215718, 4747 ins, 24510 del, 87431 sub ] exp/mono/decode_test/cer_10_0.0
* %WER 32.61 [ 70336 / 215718, 5866 ins, 16282 del, 48188 sub ] exp/tri1/decode_test/cer_13_0.0
* %WER 32.10 [ 69238 / 215718, 6186 ins, 15772 del, 47280 sub ] exp/tri2/decode_test/cer_13_0.0
* %WER 30.40 [ 65583 / 215718, 6729 ins, 13115 del, 45739 sub ] exp/tri3a/decode_test/cer_12_0.0
* %WER 27.53 [ 59389 / 215718, 6311 ins, 13008 del, 40070 sub ] exp/tri4a/decode_test/cer_15_0.0
* %WER 31.42 [ 67779 / 215718, 6565 ins, 13660 del, 47554 sub ] exp/tri4a/decode_test.si/cer_12_0.0
* %WER 24.21 [ 52232 / 215718, 6425 ins, 11543 del, 34264 sub ] exp/tri5a/decode_test/cer_15_0.0
* %WER 27.83 [ 60025 / 215718, 6628 ins, 12107 del, 41290 sub ] exp/tri5a/decode_test.si/cer_12_0.0
* %WER 17.07 [ 36829 / 215718, 4734 ins, 9938 del, 22157 sub ] exp/nnet3/tdnn_sp/decode_test/cer_12_0.0


