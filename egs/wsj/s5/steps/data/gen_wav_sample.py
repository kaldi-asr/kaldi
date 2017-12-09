#encoding:utf8
# Copyright     2017  Ye Bai
# Apache 2.0
#
# This script generates wave samples based on wav.scp.

from __future__ import print_function
import subprocess
import argparse
import re

def execute_command(command):
    p = subprocess.Popen(command, shell=True)
    p.communicate()
    if p.returncode is not 0:
        raise Exception("Command exited with status {0}: {1}".format(
                p.returncode, command))
                

def get_args():
    parser = argparse.ArgumentParser(description="Generate wave samples based on wav.scp.\n"
                                                 "Usage: gen_wav_sample.py [options...] <wav.scp> <out-data-dir>\n"
                                                 "E.g. python gen_wav_sample.py wav.scp sample",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)                
                
    parser.add_argument("--num", type=int, dest = "sample_num", default = 10, 
                                        help="Number of samples to generate.")

    parser.add_argument("--prefix", type=str, dest = "prefix", default = "", 
                                        help="Prefix of generated samples.")
                                        
    parser.add_argument("--record_file", type=str, dest = "record_file", default = "", 
                                        help="A list of records to generate samples.\n"
                                            " The format is like this:"
                                            "rvb_noise_f001_000008.wav \n"
                                            "rvb_noise_f001_000009.wav \n"
                                            "...")
                                        
    parser.add_argument("scp_file",
                        help="Input data")
    parser.add_argument("output_dir",
                        help="Output data directory")
                                        
    args = parser.parse_args()
    return args
    
if __name__ == '__main__':

    args = get_args()
    sample_num = args.sample_num
    prefix = args.prefix
    scp_file = args.scp_file
    output_dir = args.output_dir
    
    execute_command("mkdir -p {}".format(output_dir))
    
    if prefix:
        prefix += "_"
    
    record_dic = {}

    if args.record_file:
        with open(args.record_file) as reco_file:
            for line in reco_file:
                id = line.strip()
                if id not in record_dic:
                    record_dic[id] = 1
    
    with open(scp_file, "r") as f:   
        for line in f:
            if sample_num == 0:
                break
            items  = line.strip().split()
            id = items[0]
            if record_dic:
                if id not in record_dic:
                    continue
                else:
                    reco = " ".join(items[1:]).strip()
                    if reco[-1] == "|":
                        execute_command("{wav} > {outpath}/{out_name}".format(wav=reco[0:-1],
                                    outpath=output_dir,out_name=prefix+id))
                    else:
                        execute_command("wav-copy '{wav}' {outpath}/{out_name}".format(wav=reco,
                                    outpath=output_dir,out_name=prefix+id))
                    sample_num -=1
            else:                
                reco = " ".join(items[1:]).strip()
                if reco[-1] == "|":
                    execute_command("{wav} > {outpath}/{out_name}".format(wav=reco[0:-1],
                                outpath=output_dir,out_name=prefix+id))
                else:
                    execute_command("wav-copy '{wav}' {outpath}/{out_name}".format(wav=reco,
                                outpath=output_dir,out_name=prefix+id))
                sample_num -=1          
    print("Done.")












