function sPhone = iPhone2sPhone(iPhone, Cfg)

Cfg.init = 1;
if ~isfield(Cfg,'sLanguage')
  Cfg.sLanguage = 'English';
end

switch Cfg.sLanguage
  case 'English'
    Phones = {'AA','AE','AH','AO','AW','AY','B','CH','D','DH','EH','ER','EY','F','G','HH','IH','IY','JH','K','L','M','N','NG','OW','OY','P','R','S','SH','SIL','T','TH','UH','UW','V','W','Y','Z','ZH','FIL','NON','NPS','SP','SPK','STA','XXX'};
  case 'German'
    Phones = {'@','a','a~','ah','aI','aU','b','C','d','dZ','E','eh','Eh','eI','er','f','g','h','I','ih','j','k','l','m','n','N','O','o~','oe','OE','oh','oU','OY','p','pf','r','R','s','S','sil','sp','t','T','ts','tS','U','uh','v','x','xxx','Y','yh','z','Z'};
  otherwise 
    error(['This parameter is case-sensitive; I didn''t recognize ' Cfg.sLanguage '.'])
end

sPhone = Phones{iPhone};