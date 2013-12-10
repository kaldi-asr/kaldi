#!/bin/bash                                                                    
. path.sh                                                                      
data=$1                                                                        
old_ali_dir=$2                                                                 
mix_ali_dir=$3                                                                 
mkdir -p $mix_ali_dir                                                          
                                                                               
cp $old_ali_dir/{final.mdl,num_jobs,tree} $mix_ali_dir/   
                                                                               
gunzip -c $old_ali_dir/ali.*.gz | gzip -c > $old_ali_dir/ali.gz                
                                                                               
feats="ark,s,cs:copy-feats scp:$data/feats.scp ark:- |"                        
copy-clean-ali "$feats" "ark:gunzip -c $old_ali_dir/ali.gz |" "ark:| gzip -c > $mix_ali_dir/ali.1.gz"
