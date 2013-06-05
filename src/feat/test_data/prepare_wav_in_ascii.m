%convert the 16kHz, lin16, wav to ascii float format
test = wavread('test.wav');
test = test*32768;

fo = fopen('test_matlab.ascii','w');
fprintf(fo,'[',);
for i=1:size(test,1)
  fprintf(fo,' %g',test(i));
end
fprintf(fo,' ]');
fclose(fo);
