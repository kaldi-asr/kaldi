LANG=C awk '
!r && NR > 1 { if(!nmix[$2]++) npdf++ }
r {
  if(FNR == 1) {
    dim=substr($1,3)+0
    print "<DIMENSION> " dim
    print "<NUMPDFS> "   npdf
    next
  }
  gmm_name=$2
  weights = ""
  gconsts  = ""
  means   = ""
  vars    = ""
  for(i=0; i<nmix[gmm_name]; i++) {
    if(i>0) { 
      getline
      if($2 != gmm_name) {
        print "Error: GMM components are not sorted" > /dev/stderr
        exit(-1)
      }
    }
    weights = weights " " $3;
    gconsts = gconsts " " 0;
    for(j=0; j<dim; j++) {
      means = means " " $(j+4)/$(j+4+dim)
      vars  = vars  " " 1/$(j+4+dim)
    }
    means = means "\n"
    vars = vars "\n"
  }
  print "<DiagGMMBegin>"
  print "<GCONSTS> FV " nmix[gmm_name]
  print "[ " gconsts " ]"
  print "<WEIGHTS> FV " nmix[gmm_name]
  print "[ " weights " ]"
  print "<MEANS_INVVARS> FM " nmix[gmm_name] " " dim
  print "[ " means " ]"
  print "<INV_VARS> FM " nmix[gmm_name] " " dim
  print "[ " vars " ]"
  print "<DiagGMMEnd>"
}
' $1 r=1 $1