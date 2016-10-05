phone_ids_on_lattice='/Users/jinyiyang/Downloads/Matlab_Bernd/post_on_lat_lex_word_tg/hayd/1best.phones_bernd.id';
phone_posts_on_lats='/Users/jinyiyang/Downloads/Matlab_Bernd/post_on_lat_lex_word_tg/hayd/1best.pdf.post';

%phone_ids_on_lattice='/Users/jinyiyang/Downloads/Matlab_Bernd/post_from_nnet1/440c02010/440c02010.phone_bernd.id'; %post_from_nnet1
%phone_posts_on_lats='/Users/jinyiyang/Downloads/Matlab_Bernd/post_from_nnet1/440c02010/440c02010.phone.post';%post_from_nnet1

%phone_ids_on_lattice='/Users/jinyiyang/Downloads/Matlab_Bernd/post_on_lats_lex_phone_dep_bg/440c02010/5best.phone_bernd.id.1';
%phone_posts_on_lats='/Users/jinyiyang/Downloads/Matlab_Bernd/post_on_lats_lex_phone_dep_bg/440c02010/5best.pdf.post.1';
%sFilePhoneMapping = '/Users/jinyiyang/Downloads/Matlab_Bernd/phone.tab';

phone_ids=importdata(phone_ids_on_lattice);
phone_post=importdata(phone_posts_on_lats);
len_phone_post=length(phone_ids);


CLASSES_ARPABET = {'AO','AA','IY','UW','EH','IH','UH','AH','AE','EY','AY','OW','AW','OY','ER','P','B','T','D','K','G','CH','JH','F','V','TH','DH','S','Z','SH','ZH','HH','M','N','NG','L','R','Y','W','NSN','SIL','SPN'};

mLats=zeros(length(CLASSES_ARPABET),length(phone_ids));

for (k=1:len_phone_post)
    mLats(phone_ids(k),k)=phone_post(k);
end


%t = ([1:size(mLats,2)]-1)/100; 
t=length(phone_ids)/100;
y = 1:length(CLASSES_ARPABET);

imagesc(t,y,mLats);
imagesc(mLats);
set(gca,'ytick',1:length(CLASSES_ARPABET));
set(gca,'yticklabel',CLASSES_ARPABET);
title('Phone post from lats:Trigram (Word lex)');
xlabel('Time/s')
ylabel('Phoneme')