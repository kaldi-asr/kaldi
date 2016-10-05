function Labels = read_kaldi_phoneorder(sFile)
% sFile = '/Users/bernd/grid/kaldi/egs/push_forward/s5/tmp/phone_mappings/pseudo_phones.txt';
fid = fopen(sFile,'r');

D = textscan(fid,'%s %d8');
fclose(fid);
D = D{1};
Labels = strrep(D,'_B','');
