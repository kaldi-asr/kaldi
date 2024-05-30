%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% Copyright 2017 Johns Hopkins University (Author: Aswin Shanmugam Subramanian)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

function stoi_estoi_sdr(nj,enhancement_method,destination_directory,set)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% "stoi_estoi_sdr" : this function computes the average STOI, eSTOI and SDR
%                    scores by calling downloaded third party matlab functions
%
% Input:
% nj: number of jobs
% enhancement_method: the name of the enhacement method
% destination_directory: the directory where the results have to be stored,
%                        the list of the enhaced and reference files are
%                        stored here before calling this function
% set: name of the set to be evaluated ('et05' or 'dt05')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

original_file_list=strcat(destination_directory,'/original_list');
enhanced_file_list=strcat(destination_directory,'/enhanced_list');
files1=textread(original_file_list,'%s');
files2=textread(enhanced_file_list,'%s');
d_stoi=zeros(1,length(files2));
d_estoi=zeros(1,length(files2));
SDR=zeros(1,length(files2));
p = parpool('local', nj);
parfor i=1:length(files2)
    [x, fs] = audioread(files1{i});
    [y, fs] = audioread(files2{i});
    m=length(x);
    n=length(y);
    d=abs(m-n);
    if m>n
         y=[y; zeros(d,1)];
    end
    if n>m
         x=[x; zeros(d,1)];
    end

    d_stoi(i)=stoi(x,y,fs);
    d_estoi(i)=estoi(x,y,fs);
    [SDR(i),SIR,SAR,perm]=bss_eval_sources(y',x');
end
SDR_avg=mean(SDR);
STOI_avg=mean(d_stoi);
ESTOI_avg=mean(d_estoi);
SDRFile=strcat(destination_directory,'/',enhancement_method,'_',set,'_SDR');
stoiFile=strcat(destination_directory,'/',enhancement_method,'_',set,'_STOI');
estoiFile=strcat(destination_directory,'/',enhancement_method,'_',set,'_eSTOI');
fileID = fopen(SDRFile,'w');
fprintf(fileID,'%f\n',SDR_avg);
fclose(fileID);
fileID = fopen(stoiFile,'w');
fprintf(fileID,'%f\n',STOI_avg);
fclose(fileID);
fileID = fopen(estoiFile,'w');
fprintf(fileID,'%f\n',ESTOI_avg);
fclose(fileID);
ResultMATFile=strcat(destination_directory,'/',enhancement_method,'_',set,'_stoi_estoi_sdr.mat');
save(ResultMATFile,'SDR','d_stoi','d_estoi');
end
