function iPhone = sPhone2iPhone(sPhone,Cfg)
% This list should be compatible with standard sphinx, htk, and kaldi
% format
Cfg.init = 1;
if ~isfield(Cfg,'sLanguage')
  Cfg.sLanguage = 'English';
end

switch Cfg.sLanguage
  case 'English'
    sPhone = upper(sPhone);
    Phones = {'AA','AE','AH','AO','AW','AY','B','CH','D','DH','EH','ER','EY','F','G','HH','IH','IY','JH','K','L','M','N','NG','OW','OY','P','R','S','SH','SIL','T','TH','UH','UW','V','W','Y','Z','ZH','FIL','NON','NPS','SP','SPK','STA','XXX'};
    % Phones = {'aa','ae','ah','ao','aw','ay','b','ch','d','dh','eh','er','ey','f','g','hh','ih','iy','jh','k','l','m','n','ng','ow','oy','p','r','s','sh','sil','t','th','uh','uw','v','w','y','z','zh'
  case 'German'
    % The German phone list needs to be case sensitive
    Phones = {'@','a','a~','ah','aI','aU','b','C','d','dZ','E','eh','Eh','eI','er','f','g','h','I','ih','j','k','l','m','n','N','O','o~','oe','OE','oh','oU','OY','p','pf','r','R','s','S','sil','sp','t','T','ts','tS','U','uh','v','x','xxx','Y','yh','z','Z'};
end
iPhone = findcell(Phones,sPhone);
if iPhone == 0 | isempty(iPhone)
  error(['I don''t know what to do with phone ' sPhone '.'])
end

% -------------------------------------------------------------------------
function vIdx = findcell(C,s,sMode)
% VIDX= FINDCELL(C,S)
% C: cell array
% s: search string
% vIdx: index vector of cells containing that string at least once
% If s is a 1-dim cell array, findcell returns 1 for each element in C that
% contains *all* strings stored in s (see example below)
% Bernd Meyer, July 2009
% ------------------------
% A = {'asdf','qwer','yxcv','bier','mama'}; vIdx = findcell(A, 'bier');
% vIdx -> 4
% A = {'asdf','qwer','yxcv','bier','mama'}; vIdx = findcell(A, 'a')
% vIdx = 1 5
%
% If s is a cell array:
% findcell({'abcdef','abcde','abcdef'},{'abc','def'})
%  --> vIdx is [1 3]
if ~exist('sMode','var')
  sMode = 'strict'; % strict or loose
end


if ischar(s)
  if strcmp(sMode,'loose')
    idx_temp = strfind(C,s); vIdx = find(not(cellfun('isempty', idx_temp)));
  elseif strcmp(sMode,'strict')
    vIdx = zeros(1,length(C));
    for k = 1:length(C)
      if strcmp(C{k},s) % strcmp instead of strfind makes it strict
        vIdx(k) = 1;
      end
    end
    vIdx = find(vIdx);
  else
    error(['Mode should be strict or loose, but is ' sMode]);
  end
elseif iscell(s)
  if strcmp(sMode,'strict'); error('not supported'); end
  vIdx = zeros(1,length(C));
  for k = 1:length(C)
    vFound = zeros(1,length(s));
    for m = 1:length(s)
      if strfind(C{k},s{m})
        vFound(m) = 1;
      end
    end
    if isempty(find(vFound == 0,1))
      vIdx(k) = 1;
    end
  end
  vIdx = find(vIdx);
else
  error('Input argument s should be either a character/string or a cell array');
end
