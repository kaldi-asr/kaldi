function [mFeat,header] = load_htk(infile,display_flag,byteswap_flag)

if ~exist('byteswap_flag','var')
  byteswap_flag = 1; 
end

if ~exist('display_flag','var')
  display_flag = 0; 
end

if ~exist('infile','var')
  D = dir('*.htk');
  if ~isempty(D)
    infile = D(1).name; 
    disp(['Trying to read any htk file from this directory...found ' infile])
  else
    disp('Trying to read any htk file from this directory. Didn''t find one.')
  end
end

% open file
if byteswap_flag
   fid=fopen(infile,'r','b');
else
   fid=fopen(infile,'r','l');
end;

[header,indata] = read_head_and_data(fid);
% close file
fclose(fid);

dim=header.sampSize/4;

% rearrange vector to matrix
if ~isequal(dim*header.nSamples,length(indata))
  % most likely the byteordering was wrong -> reopen the file
  % with b and l swapped.
  if byteswap_flag
     fid=fopen(infile,'r','l');
  else
     fid=fopen(infile,'r','b');
  end
  [header,indata] = read_head_and_data(fid);
  disp('Warning: The byteorder flag in load_htk isn''t set correctly, which will slow things down')
end

% calculate feature dimension
dim=header.sampSize/4;

mFeat = reshape(indata,dim,header.nSamples);    

if display_flag
   
  % Plot
  
  y=1:1:dim;                
  len=header.nSamples*header.sampPeriod/10^7;
  x=0:header.sampPeriod/10^7:len;          
  imagesc(x, y, mFeat);%, [-0.5 1]);
  axis xy;
  colormap jet;
  set(gca,'FontSize', 20); 
  cb=colorbar;
  set(cb,'FontSize', 20); 
  
  xlabel('time in s');
  ylabel('# feature')
  
end
% -------------------------------------------------------------------------
function [header,indata,count] = read_head_and_data(fid)
% read header
[header.nSamples]=fread(fid,1,'int'); 
[header.sampPeriod]=fread(fid,1,'int'); 
[header.sampSize]=fread(fid,1,'short'); 
[header.sampKind]=fread(fid,1,'short');

% read data
[indata, count]=fread(fid,inf,'float'); 
