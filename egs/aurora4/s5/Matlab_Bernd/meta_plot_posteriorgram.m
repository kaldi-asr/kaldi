% function meta_plot_posteriorgram
% Use this to visualize monophone activations obtained from your favorite
% speech file and neural net.
%
% This requires:
% pseudo_phones.txt (from Harish's tool, currently just eats the CMU dict phones)
% yoursignal.wav (16kHz mono signal)
% yoursignal.monophone (monophone activations generated with Harish's tools
%                       in HTK format)
% Bernd Meyer, Apr-2016
clear
close('all')

% Set these parameters to something that works for you:
sFilePhoneMapping = '/Users/jinyiyang/Downloads/Matlab_Bernd/phone.tab';
sWavFile = '/Users/jinyiyang/Downloads/Matlab_Bernd/post_from_nnet1/morning/good_morning.wav';
sMonoActivations = '/Users/jinyiyang/Downloads/Matlab_Bernd/post_from_nnet1/morning/good_morning.monophone';

Labels = read_kaldi_phoneorder(sFilePhoneMapping);
MPgram = load_htk(sMonoActivations);

% randn(length(Labels),250);
% MPgram(1,:) = ones(1,250); % fill one row with ones to keep track of
% sorting and merging

% Reorder the phone set
vOrder = label_reorder_arpabet(Labels); % ,'Phonemes');
MPgram_ordered = MPgram(vOrder,:);
Labels_ordered = Labels(vOrder);

% Collapse into phonetic classes (fricative, stop, affricative, ...)
[MPgram_pclass,Labels_pclass] = collapse_posteriorgrams(MPgram_ordered,'pclass');

% Collapse into vowel/consonant/other
[MPgram_cv,Labels_cv] = collapse_posteriorgrams(MPgram_ordered,'cv');

vSig=audioread('/Users/jinyiyang/Downloads/Matlab_Bernd/post_from_nnet1/hynek_test/hynek.wav');
figure(123)
subplot(2,2,1)
specgram(vSig)
title('Spectrogram')

subplot(2,2,2)
plot_posteriorgram(MPgram_ordered,Labels_ordered);
title('All phones, ARBABET sorting')

subplot(2,2,3)
plot_posteriorgram(MPgram_pclass,Labels_pclass);
title('Collapsed view on posteriorgram')

subplot(2,2,4)
plot_posteriorgram(MPgram_cv,Labels_cv);
title('Collapsed view (vowels/consonants)')

% SIL 0
% SPN 1
% NSN 2
% S_B 3
% UW_B 4
% T_B 5
% N_B 6
% K_B 7
% Y_B 8
% Z_B 9
% AO_B 10
% AY_B 11
% SH_B 12
% W_B 13
% NG_B 14
% EY_B 15
% B_B 16
% CH_B 17
% OY_B 18
% JH_B 19
% D_B 20
% ZH_B 21
% G_B 22
% UH_B 23
% F_B 24
% V_B 25
% ER_B 26
% AA_B 27
% IH_B 28
% M_B 29
% DH_B 30
% L_B 31
% AH_B 32
% P_B 33
% OW_B 34
% AW_B 35
% HH_B 36
% AE_B 37
% R_B 38
% TH_B 39
% IY_B 40
% EH_B 41