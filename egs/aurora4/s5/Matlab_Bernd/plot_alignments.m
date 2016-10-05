% Used to plot alignments from HMM forced alignment
% sMainDir needs to be updated to make this work on your machine
% Bernd T Meyer, July 2015
% --> Transcript is encoded in filename (peter_kauft_acht and so on)
clear

Cfg.sLanguage = 'German';
sSecondaryPlot = 'spectrogram'; % spectrogram mfcc

sMainDir = '/Users/bernd/exp/alignment_htk';
sSignalDir = [sMainDir '/signals'];
sFeatDir = [sMainDir '/features'];
sTranscriptDir = [sMainDir '/transcripts'];
sAlignmentDir = [sMainDir '/alignments'];

Files = getfilenames(sAlignmentDir,'*.align');
disp(['Found ' num2str(length(Files)) ' alignments. Plotting a random alignment.'])
rng('shuffle');
vRand = randperm(length(Files));

% sFileAlign = Files{vRand(1)};
sFileAlign = Files{vRand(1)};
sFileSig = strrep(strrep(sFileAlign,sAlignmentDir,sSignalDir),'.align','.wav');
sFileFeat = strrep(strrep(sFileAlign,sAlignmentDir,sFeatDir),'.align','.htk');

[vSig,fs] = audioread(sFileSig);
disp(['Reading from ' sFileAlign])
[PData,ListPhonemes] = read_label(sFileAlign, Cfg);

% Preparing the plot: This should produce a feature matrix plot with the
% correct phoneme labels on the time axis. For readability, each label
% should be plotted only when the phoneme begins. Turns out, it is - at
% least for this sentence - not very readable.
iNumLabels = PData{end,2}; % number of different phonemes in the labels
C = cell2mat(PData(:,2)); % frame-wise phoneme counter
idxLabelChange = find(diff(cell2mat(PData(:,2)))); % start frame of new
idxLabelChange = [1;idxLabelChange]; % phonemes (incl. first frame)

figure(234)
subplot(211)
t = [0:length(vSig)-1]/fs;
plot(t,vSig)
hold on
set(gca,'XTick',idxLabelChange/100);
set(gca,'XTickLabel',ListPhonemes);
for k = 1:length(idxLabelChange)
  plot([idxLabelChange(k)/100 idxLabelChange(k)/100],[-1 1],'k')
end

hold off
title(strrep(sFileSig,'_',' '))
axis tight

subplot(212)
switch sSecondaryPlot
  case 'spectrogram'
    spectrogram(vSig,400,300,[],16000,'yaxis')
    hold on
    
    set(gca,'xtick',idxLabelChange/100);
    set(gca,'xticklabel',ListPhonemes);
    hold off
  case 'mfcc'
    MFeat = load_htk(sFileFeat);
    MFeat = MFeat(1:13,:);    
    imagesc(MFeat)
    set(gca,'xtick',idxLabelChange);
    set(gca,'xticklabel',ListPhonemes);
  otherwise
    disp(['Unknown kind of plot: ' sSecondaryPlot '. Try spectrogram or mfcc.'])
end
