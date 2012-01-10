#!/bin/awk -f
BEGIN {
  if(dim=0) {
    print "./mdl_change_feature_dim.awk dim=N MODEL";
    exit 1;
  }
  active=0
}

/^<DIMENSION>/ {
  str= $1" "dim
  for(i=3;i<=NF;i++) {
    str = str" "$i
  }
  print str
  next;
}

/^<INV_VARS>/ { active=1 }
/^<MEANS_INVVARS>/ { active=1 }

{
  if(!active || "[" == substr($0,length($0),1)) {
    print $0;
  } else {
    str="1"
    for(i=2;i<=dim;i++) { str=str" 1"; }
    if("]" == substr($0,length($0),1)) {
      str=str" ]";
      active=0;
    }
    print str
  }
}




