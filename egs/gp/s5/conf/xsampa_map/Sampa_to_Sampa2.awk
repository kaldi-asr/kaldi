# Copyright 2012  Milos Janda

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.

#!/bin/awk -f

{
    for (i=2; i<=NF; i++) {	
	gsub("^0","_0",$i);
	gsub("^1","_1",$i);
	gsub("^2","_2",$i);
	gsub("^3","_3",$i);
	gsub("^4","_4",$i);
	gsub("^5","_5",$i);
	gsub("^6","_6",$i);
	gsub("^7","_7",$i);
	gsub("^8","_8",$i);
	gsub("^9","_9",$i);
	
	gsub("@","_at",$i);
	gsub("&","_amp",$i);
	gsub("\\\\","_backs",$i);
	gsub("{","_<",$i); 
	gsub("}","_>",$i);

	# post-processing :)
	#gsub("__","_",$i);
    }
    print
}
