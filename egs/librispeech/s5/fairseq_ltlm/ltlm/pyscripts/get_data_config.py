# Copyright 2021 STC-Innovation LTD (Author: Anton Mitrofanov)
import argparse
import json
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', nargs="+", required=True,
                        help="Train.  format 'lats,[suff:default empty],[epoch:default 0] lats2 lat3,suff,epoch'")
    parser.add_argument('--valid', nargs="+", required=True, help="Valid. format 'lats,ref lat2,ref2'")
    parser.add_argument('--test', nargs='*', default=None, help="Test. format like valid")
    parser.add_argument('--split_per_epoch', default=3, type=int, help='Numbers of splits for every epoch')
    parser.add_argument("--out", type=str, required=True, help="Path to result data_config.json file")
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    
    last_e=0
    last_e_splits=0
    data = {"train": [], }
    for l_s_e in args.train:
        l_s_e_sep = l_s_e.split(',')
        l = l_s_e_sep[0]
        s = l_s_e_sep[1] if len(l_s_e_sep) > 1 else ''
        e = l_s_e_sep[2] if len(l_s_e_sep) > 2 else ''
        if e:
            e=int(e)
            last_e=max(e+1, last_e)
        else:
            if last_e_splits >= args.split_per_epoch:
                last_e+=1
                last_e_splits=0
            e=last_e
            last_e_splits+=1
            
        data['train'].append({"epoch": e, "lats": l, "utt_suff": s})

    data["valid"] = []
    for l_r in args.valid:
        l_r_sep = l_r.split(',')
        assert len(l_r_sep) == 2, RuntimeError("Wrong valid format. Must be lat,ref lat2,ref ...")
        l, r = l_r_sep
        data['valid'].append({'lats': l, 'ref': r})
    if args.test:
        data['test'] = []
        for l_r in args.test:
            l_r_sep = l_r.split(',')
            assert len(l_r_sep) == 2, RuntimeError("Wrong test format. Must be lat,ref lat2,ref ...")
            l, r = l_r_sep
            data['test'].append({'lats': l, 'ref': r})
    with open(args.out, 'w') as out:
        json.dump(data, out, indent=4)


if __name__ == "__main__":
    main()
