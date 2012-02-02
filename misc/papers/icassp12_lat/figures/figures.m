# run this in octave




%beam = [15.0         15.0       15.0       15.0       15.0         15.0        15.0        15.0        15.0         15.0]

latbeam = [10.0          9.0        8.0        7.0        6.0          5.0         4.0         3.0         2.0          1.0 ];
density =  [23.56        16.0       10.82       7.26       4.91         3.38        2.38        1.77        1.4          1.17];
realtime = [2.317        2.114      2.006      1.941      1.900        1.855       1.835       1.797       1.769        1.737];
baseline_err = [  11.51        11.51      11.51      11.51      11.51        11.51       11.51       11.51       11.51        11.51 ];
baseline_num_err =[  227          227        227         227        227          227          227        227          227         227];
oracle = [  2.62         2.78       2.85       3.08       3.35         3.86        4.52        5.46        6.98         8.62 ];
num_err = [93           96         96         101        105          116         129         147         168          191];
rescore_err =[ 9.59         9.59       9.59       9.59       9.56         9.59        9.61        9.70        9.80        10.21  ];
rescore_utt_wrong = [205          205          205        205       204          204         204         202         205          212 ];;

labelsz = 27;
figure(1)
hold off

set(1,"Defaulttextfontsize",labelsz) 
set(1,"Defaultaxesfontsize",labelsz) 

subplot(2,2,1)
plot(latbeam, density)
xlabel('Lattice beam');
ylabel('Lattice density');
subplot(2,2,2)
plot(latbeam, baseline_err, 'kx');
hold on
plot(latbeam, oracle);
hold off
set(gca(), "ylim", [2.5, 14.0]);
legend('One-best WER','Oracle WER')
xlabel('Lattice beam');
ylabel('WER');
subplot(2, 2, 3);
plot(latbeam, rescore_err);
xlabel('Lattice beam');
ylabel('WER, rescoring with trigram LM');
subplot(2, 2, 4);
plot(latbeam, realtime);
xlabel('Lattice beam');
ylabel('Real time factor');

print -F:23  -deps 'latbeam.eps'



decode_beam = [16.0        15.0          14.0       13.0      12.0         11.0         10.0       9.0          8.0];
%lattice beam: 7.0         7.0           7.0        7.0       7.0          7.0          7.0       7.0          7.0 0
density = [ 7.4         7.26          7.04       6.74      6.33         5.71         4.9       4.08         3.09 ];
realtime = [ 2.646       1.941         1.428      1.052     0.775        0.555        0.388     0.267        0.177 ];
wer = [ 11.51        11.51        11.54      11.58     11.75        11.93        12.32      13.47        15.41 ];
%227          227          227        227       228          228          229        236          246
oracle = [ 3.03         3.08         3.24       3.37      3.56         4.02         4.72       6.45         9.18 ];
%100          101          105        108       112          119          128        153          184
rescore = [ 9.59         9.59         9.73       9.80      10.05        10.26        10.55      11.82        13.79  ];
%205          205          206        206       207          209          208        220          235


subplot(2,2,1)
plot(decode_beam, density)
xlabel('Decoding beam');
ylabel('Lattice density');
subplot(2,2,2)
plot(decode_beam, wer, 'kx');
hold on
plot(decode_beam, oracle);
legend('One-best WER', 'Oracle WER')
xlabel('Decoding beam');
ylabel('WER');
hold off
subplot(2, 2, 3);
plot(decode_beam, wer, 'kx');
hold on
plot(decode_beam, rescore);
legend('One-best WER', 'Rescored WER')
xlabel('Decoding beam');
ylabel('WER, rescoring with trigram LM');
hold off
subplot(2, 2, 4);
plot(decode_beam, realtime);
xlabel('Decoding beam');
ylabel('Real time factor');

print -F:23  -deps 'decodebeam.eps'



