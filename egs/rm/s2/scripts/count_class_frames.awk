#!/bin/awk -f
BEGIN {
  
  if (ARGC != 3) {
    print "count_class_frames.awk rfile wfile"; exit 1;
  }
  
  if (ARGV[1] == "-") {
    f_in = "/dev/stdin";
  } else {
    f_in = ARGV[1];
  }
  if (ARGV[2] == "-") {
    f_out = "/dev/stdout";
  } else {
    f_out = ARGV[2];
  }

  max_ii=0;
  while(getline < f_in) {
    #print NF $0
    for(ii=2;ii<=NF;ii++) {
      lab = int($ii)
      counts[lab]++;
      if(lab > max_ii) { max_ii = lab; }
    }
  }

  printf("[") > f_out;
  for (ii=0; ii<=max_ii; ii++) {
    printf(" %d", counts[ii]) > f_out;
  }
  printf(" ]\n") > f_out

}
