function plot_posteriorgram(MData,Labels)
t = ([1:size(MData,2)]-1)/100; 
y = 1:length(Labels);

imagesc(t,y,MData)
set(gca,'ytick',1:length(Labels))
set(gca,'yticklabel',Labels)
% set(gca,'xticklabel',);
xlabel('Time / s')
ylabel('Phoneme')
