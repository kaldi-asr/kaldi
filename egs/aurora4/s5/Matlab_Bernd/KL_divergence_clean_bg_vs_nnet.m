%%%/usr/local/bin/matlab
%!/usr/local/R2015a
% KL Divergence
% D(P|Q)=sigma{P(i)log[(P(i)/Q(i)]}

num_frames=241125;
num_phones=42;

%name_of_data=
% %post_from_lats_lex_phone_bg=importdata('/Users/jinyiyang/Downloads/Matlab_Bernd/post_on_lats_lex_phone_dep_bg/frodo/1best.phones.post.matrix');

%frodo
% post_from_lats_lex_word_tg=fopen('/Users/jinyiyang/Downloads/Matlab_Bernd/post_on_lat_lex_word_tg/frodo/1best.phones.post.matrix');
% post_from_lats_lex_phone_bg=fopen('/Users/jinyiyang/Downloads/Matlab_Bernd/post_on_lats_lex_phone_dep_bg/frodo/1best.phones.post.matrix');
%post_from_nnet1=fopen('/Users/jinyiyang/Downloads/Matlab_Bernd/post_from_nnet1/frodo/frodo.nnet1.phone.post.matrix_bernd_cp');

%how are you doing
% post_from_lats_lex_word_tg=fopen('/Users/jinyiyang/Downloads/Matlab_Bernd/post_on_lat_lex_word_tg/hayd/1best.phones.post.matrix');
% post_from_lats_lex_phone_bg=fopen('/Users/jinyiyang/Downloads/Matlab_Bernd/post_on_lats_lex_phone_dep_bg/hayd/1best.phones.post.matrix');
% post_from_nnet1=fopen('/Users/jinyiyang/Downloads/Matlab_Bernd/post_from_nnet1/hayd/hayd.nnet1.phone.post.matrix_bernd_cp');


%440c02010
% post_from_lats_lex_word_tg=fopen('/Users/jinyiyang/Downloads/Matlab_Bernd/post_on_lat_lex_word_tg/440c02010/5best.lats.phonemes_bernd.post.matrix.1');
% post_from_lats_lex_phone_bg=fopen('/Users/jinyiyang/Downloads/Matlab_Bernd/post_on_lats_lex_phone_dep_bg/440c02010/5best.phone_bernd.post.matrix.1');
% post_from_nnet1=fopen('/Users/jinyiyang/Downloads/Matlab_Bernd/post_from_nnet1/440c02010/440c02010.phone_bernd.post.matrix');


%eval92_clean(40min)
post_from_lats_lex_word_tg=fopen('/export/a07/jyang/kaldi_harish/egs/aurora4_jinyi/s5/post_on_lattice_nnet1_multi/eval92_clean/eval92_clean.post.matrix');
post_from_lats_lex_phone_bg=fopen('/export/a07/jyang/kaldi_harish/egs/aurora4_jinyi/s5/post_on_lattice_nnet1_multi_phone_lex_bg_dep/eval92_clean/eval92_clean.post.matrix');
post_from_nnet1=fopen('/export/a07/jyang/kaldi_harish/egs/aurora4_jinyi/s5/post_from_nnet1_multi/test_eval92_clean/monophone_bnfea_test_eval92_clean.all.matrix');


P=post_from_lats_lex_phone_bg;
%Q=post_from_lats_lex_phone_bg;
Q=post_from_nnet1;

line_P=fgetl(P);
line_Q=fgetl(Q);

PP = zeros(num_frames,num_phones);
QQ = zeros(num_frames,num_phones);
flag = 0;
while ischar(line_P)
    
    flag = flag + 1;
    array_P=strsplit(line_P);
    array_Q=strsplit(line_Q);
    %1st one is ''
    if length(array_Q)== num_phones
        QQ(flag,1:num_phones)=str2double(array_Q(1:num_phones));
    else
    QQ(flag, 1:num_phones) = str2double(array_Q(2:num_phones+1));
    end
    if length(array_P)== num_phones
        PP(flag,1:num_phones)=str2double(array_P(1:num_phones));
    else
    PP(flag, 1:num_phones) = str2double(array_P(2:num_phones+1));
    end
  %  for n=1:length(array_P)
%        if strcmp(array_P{1},'') || strcmp(array_P{1},' ')
%            array_P{1}=;
    %    end
  %  end
  line_P=fgetl(P);
  line_Q=fgetl(Q);
end
P=PP;
Q=QQ;



fprintf('P size is %d %d vs Q size is %d %d\n',size(P),size(Q));
%fprintf('length of post_lats is %.f\n',length(post_from_lats));
if size(P) ~= size(Q) 
    fprintf ('Dim mismatch: P:%d * %d VS. Q:%d * %d', P,Q);
    quit cancel;
end

S=zeros(size(P));
D=zeros(1,num_frames);

S=P.*log(P./(Q+eps));

D=sum(S,2); % 2nd dim
average=sum(D)/num_frames;
fprintf ('Sum is %f\n',average);

Q_max=max(Q,[],2);
Q0=zeros(size(Q_max));
S0=zeros(size(D));
for i=1:13
    Q0=Q0+Q_max;
    S0=S0+Q_max.*D;    
end

D_new=S0./(Q0+eps);
average_new=sum(D_new)/num_frames;
fprintf ('Smoothing sum is %f\n',average_new);


y = (1:length(D_new))/100;

save('KL_clean_tg_vs_nnet.txt','D_new');
plot(y,D_new);
title('Post from lats:trigram, word lex vs. bigram, phone lex');
ylabel('KL divergence');
xlabel('Time/s');
savefig('KL_clean_bg_nnet.fig');
%saveas(gcf,'KL_clean_bg_nnet','jpg');
%h=figure('visible','off');
