#!/bin/awk -f

# read whole DCT file
{
	for (i=2; i<=NF; i++){DCT[$i]++;}		
}
END{
# print output file
	for (i in DCT){
		#print i " - " DCT[i];
		if (i=="SIL") continue; # throw-away SIL phone
		print i;
	}
}