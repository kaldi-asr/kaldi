# Copyright 2021 STC-Innovation LTD (Author: Anton Mitrofanov)
import argparse
import os

def make_kadli_dir(dirname, text):
    print(f"Making {dirname}")
    os.makedirs(dirname, exist_ok=True)
    with open(os.path.join(dirname, 'text'), 'w', encoding='utf-8') as f_out:
        f_out.write(''.join(text))


if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--utts_per_split", default=200000, type=int)
    parser.add_argument("--add_utt_ids", action='store_true') 
    parser.add_argument("--utt_pref", default='extra_')
    parser.add_argument('texts', nargs='+')

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    curr_i=0
    buff = []
    for file_i, text in enumerate(args.texts):
        with open(text, 'r', encoding='utf-8') as f:
            for line_i, line in enumerate(f):
                if not line.strip():
                    continue
                if args.add_utt_ids:
                    line = f"{args.utt_pref}{file_i}_{line_i} {line}" 

                buff.append(line)
                if len(buff) >= args.utts_per_split :
                    make_kadli_dir(os.path.join(args.out_dir, os.path.basename(args.out_dir) + "_" + str(curr_i)), buff)
                    curr_i+=1
                    buff=[]

    if len(buff) > 0 : 
        make_kadli_dir(os.path.join(args.out_dir, os.path.basename(args.out_dir) + "_" + str(curr_i)), buff)
