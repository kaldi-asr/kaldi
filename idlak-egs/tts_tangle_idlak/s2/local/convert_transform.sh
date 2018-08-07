# We should first reverse the order of the transformation, then convert them intelligently!
# The nnet has to be in ascii form

awk 'BEGIN{
   ntrans = 0;
}
($1 ~ /^</){
   ntrans++; dimIn[ntrans] = $3; dimOut[ntrans] = $2; learnrate[ntrans] = -1;  w = 1;
   if ($1 ~ /<[Ss]plice>/){mode[ntrans] = 1;}
   else if ($1 ~ /<Nnet>/){ntrans--; w = 0}
   else if ($1 ~ /<!EndOfComponent>/){ntrans--; w = 0}
   else if ($1 ~ /<LearnRateCoef>/){ntrans--; learnrate[ntrans] = $2; $1 =""; $2 = "";}
   else if ($1 ~ "</Nnet>"){ntrans--; w = 0}
   else if ($1 ~ /<[Aa]dd[Ss]hift>/){mode[ntrans] = 2;}
   else if ($1 ~ /<[Rr]escale>/){mode[ntrans] = 3;}
   
   else { print "Unsuported transform:", $1; exit(-2) }
}
{  if (w)
     data[ntrans] = $0;
}
END{
for (i = ntrans; i >= 1; i--) {
   l = split(data[i], v)
   if (mode[i] == 1) {
      printf "<Copy> %d %d\n", dimIn[i], dimOut[i];
      iStart =  dimIn[i] * ((l - 3) / 2) + 1;
      iEnd = iStart + dimIn[i] - 1;
      if (learnrate[i] != -1) {
         printf "<LearnRateCoef> %f ",  learnrate[i];
      }
      printf "[";
      for (j = iStart; j <= iEnd; j++) printf " %d", j;
      printf " ]\n";
      print "<!EndOfComponent>"
   }
   else if (mode[i] == 2) {
      printf "<AddShift> %d %d\n", dimOut[i], dimIn[i];
      if (learnrate[i] != -1) {
         printf "<LearnRateCoef> %f ",  learnrate[i];
      }
      printf "[";
      for (j = 2; j <= l - 1; j++) printf " %f", 0.0 - v[j];
      printf " ]\n";
      print "<!EndOfComponent>"
   }
   else if (mode[i] == 3) {
      printf "<Rescale> %d %d\n", dimOut[i], dimIn[i];
      if (learnrate[i] != -1) {
         printf "<LearnRateCoef> %f ",  learnrate[i];
      }
      printf "[";
      for (j = 2; j <= l - 1; j++) printf " %f", 1.0 / v[j];
      printf " ]\n";
      print "<!EndOfComponent>"
   }
}
}'
