#!/bin/bash


export LANG=C
export LC_ALL=C

if [ "$3" == "" ]; then
  echo "usage: $0 rnnlm.model vocab kaldi.model"
  exit
fi

v=$2
km=$3

rm -f $v

cat $1 | awk '
BEGIN{
  hix=0;
}
/vocabulary size:/{
  V1_size=$3;next;
}
/hidden layer size:/{
  h_size=$4;next;
}
/output layer size:/{
  W2_size=$4;next;
}

/Vocabulary:/{
  for (i=0;i<V1_size;i++) {getline;print $1"\t"$3>>"'$v'";voc[$1]=$3;cl[$1]=$4}; next;
}

/Hidden layer activation:/{
  for (i=0;i<h_size;i++) { getline; h[hix++]=$1; } next;
}

/Weights 0->1:/{
  for (i=0;i<h_size;i++) for (j=0;j<V1_size+h_size;j++) {
    getline;if (j<V1_size) V1[i,j]=$1;else U1[i,j-V1_size]=$1;
  }; next;
}

/Weights 1->2:/{
  for (i=0;i<W2_size;i++) for (j=0;j<h_size;j++) {
    getline;if (j<V1_size) W2[i,j]=$1;
  }; next;
}

END{
  printf "<rnnlm_v2.0> <v1> [";
  # print V1
  for (j=0;j<V1_size;j++) {
  for (i=0;i<h_size;i++) {
    printf " "V1[i,j]
  }
  print ""
  }
  # print U1
  print " ]";printf " <u1> [";
  for (j=0;j<h_size;j++) {
  for (i=0;i<h_size;i++) {
    printf " "U1[i,j]
  }
  print ""
  }
  # print b1
  print " ]";printf " <b1> [";
  for (i=0;i<h_size;i++)printf " 0.0000";
  print " ]";printf "<w2> [";
  # print w2
  for (j=0;j<h_size;j++) {
  for (i=0;i<V1_size;i++) {
    printf " "W2[i,j]
  }
  print ""
  }
  # print b2
  print " ]";printf " <b2> [";
  for (i=0;i<V1_size;i++)printf " 0.0000";
  print " ]";printf "<cl> [";
  # print cl
  for (j=0;j<h_size;j++) {
  for (i=V1_size;i<W2_size;i++) {
    printf " "W2[i,j];
  }
  print ""
  }
  # print cl_b
  print " ]";printf " <cl_b> [";
  for (i=V1_size;i<W2_size;i++) printf " 0.0000";
  print " ] <classes> [ ";
  for (i=0;i<V1_size;i++) printf " "cl[i];
  print " ]";printf " <words> ";
  for (i=0;i<V1_size;i++) print voc[i];
}' > $km
