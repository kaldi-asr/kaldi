function [PData,ListPhonemes] = read_label(sFileAlign, Cfg)
% PData 
vTime1 = [];
vTime2 = [];
ListPhonemes = cell(0);

fid = fopen(sFileAlign,'r');
while ~feof(fid)
  s = fgetl(fid);
  if strfind('0123456789',s(1))
    idx = strfind(s,' ');
    % "+1" at the end of the following line, since the labels range from 0
    % to N (number of frames), so there are N+1 labels for N frames.
    % Secondly, each frame occurs twice (e.g.: 0 18 sil, 18 24 p) which is
    % not what you want for framewise labels
    iTime1 = str2double(s(1:idx(1)-1))/10^5+1;
    iTime2 = str2double(s(idx(1)+1:idx(2)-1))/10^5;
    Phone = s(idx(2)+1:end);
    if iTime1 < iTime2+1
      % disp([num2str(vTime1) ' ' num2str(vTime2) ' ' ListPhonemes])
      vTime1(end+1) = iTime1;  %#ok<AGROW>
      vTime2(end+1) = iTime2;   %#ok<AGROW>
      ListPhonemes{end+1} = Phone;   %#ok<AGROW>
    end
  end
end
fclose('all');

PData = zeros(floor(vTime2(end)),3);
PData(:,1) = [1:floor(vTime2(end))]'; %#ok<NBRAK>
for k = 1:length(vTime1)
  PData(vTime1(k):vTime2(k),2) = sPhone2iPhone(ListPhonemes{k}, Cfg);
end

PData = num2cell(PData);
for k = 1:size(PData,1)
  
  iDummy = PData{k,2};
  PData{k,3} = iPhone2sPhone(iDummy, Cfg);
end
